package llmproxy

import (
	"encoding/json"
	"strings"

	"github.com/clawvisor/clawvisor/internal/runtime/conversation"
)

type SyntheticApprovalHistoryStripRequest struct {
	Provider                       conversation.Provider
	Body                           []byte
	AllowAnthropicCompatModelPatch bool
}

type SyntheticApprovalHistoryStripResult struct {
	Body     []byte
	Modified bool
}

const ToolApprovalSubstitutedPromptMarker = "Clawvisor paused this tool call for approval."

// StripSyntheticApprovalHistory removes Clawvisor-generated approval UI from
// conversation history before it is sent back to the upstream model. The live
// pending-approval cache is the source of truth; historical assistant text that
// looks like an approval prompt is untrusted model context and can be copied or
// hallucinated by the model on later turns.
func StripSyntheticApprovalHistory(req SyntheticApprovalHistoryStripRequest) (SyntheticApprovalHistoryStripResult, error) {
	if len(req.Body) == 0 {
		return SyntheticApprovalHistoryStripResult{Body: req.Body}, nil
	}
	body := req.Body
	modified := false
	if req.Provider == conversation.ProviderAnthropic {
		if req.AllowAnthropicCompatModelPatch {
			strippedBody, ok, err := stripAnthropicCacheControl(body)
			if err != nil {
				return SyntheticApprovalHistoryStripResult{Body: body}, err
			}
			if ok {
				body = strippedBody
				modified = true
			}
		}
		res, err := stripAnthropicSyntheticApprovalHistory(body)
		if err != nil {
			return SyntheticApprovalHistoryStripResult{Body: body}, err
		}
		if res.Modified {
			body = res.Body
			modified = true
		}
		return SyntheticApprovalHistoryStripResult{Body: body, Modified: modified}, nil
	}
	return SyntheticApprovalHistoryStripResult{Body: body}, nil
}

func stripAnthropicCacheControl(body []byte) ([]byte, bool, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return body, false, nil
	}
	model, _ := raw["model"].(string)
	if model != "" && !strings.Contains(strings.ToLower(model), "claude") {
		modified := false
		if _, ok := raw["thinking"]; ok {
			delete(raw, "thinking")
			modified = true
		}
		if stripAnthropicCompatCacheControl(raw["system"]) {
			modified = true
		}
		if stripAnthropicCompatMessageCacheControl(raw["messages"]) {
			modified = true
		}
		if !modified {
			return body, false, nil
		}
		encoded, err := json.Marshal(raw)
		if err != nil {
			return body, false, err
		}
		return encoded, true, nil
	}
	return body, false, nil
}

func stripAnthropicCompatCacheControl(val any) bool {
	switch v := val.(type) {
	case []any:
		modified := false
		for _, item := range v {
			if block, ok := item.(map[string]any); ok {
				if _, exists := block["cache_control"]; exists {
					delete(block, "cache_control")
					modified = true
				}
			}
		}
		return modified
	case map[string]any:
		if _, exists := v["cache_control"]; exists {
			delete(v, "cache_control")
			return true
		}
	}
	return false
}

func stripAnthropicCompatMessageCacheControl(val any) bool {
	msgs, ok := val.([]any)
	if !ok {
		return false
	}
	modified := false
	for _, msg := range msgs {
		m, ok := msg.(map[string]any)
		if !ok {
			continue
		}
		if stripAnthropicCompatCacheControl(m["content"]) {
			modified = true
		}
	}
	return modified
}

func stripAnthropicSyntheticApprovalHistory(body []byte) (SyntheticApprovalHistoryStripResult, error) {
	if !strings.Contains(string(body), "Clawvisor") {
		return SyntheticApprovalHistoryStripResult{Body: body}, nil
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return SyntheticApprovalHistoryStripResult{Body: body}, nil
	}
	rawMessages, ok := raw["messages"]
	if !ok {
		return SyntheticApprovalHistoryStripResult{Body: body}, nil
	}
	var messages []map[string]json.RawMessage
	if err := json.Unmarshal(rawMessages, &messages); err != nil {
		return SyntheticApprovalHistoryStripResult{Body: body}, nil
	}
	out := make([]map[string]json.RawMessage, 0, len(messages))
	modified := false
	skipNextBareApprovalReply := false
	for _, msg := range messages {
		role := rawMessageString(msg["role"])
		contentText := flattenAnthropicTaskReplyText(msg["content"])
		if skipNextBareApprovalReply {
			skipNextBareApprovalReply = false
			if role == "user" && isBareSyntheticApprovalReply(contentText) {
				modified = true
				continue
			}
		}
		if role == "assistant" && isSyntheticApprovalPromptText(contentText) {
			modified = true
			skipNextBareApprovalReply = true
			continue
		}
		out = append(out, msg)
	}
	if !modified || len(out) == 0 {
		return SyntheticApprovalHistoryStripResult{Body: body}, nil
	}

	// Merge consecutive messages with the same role (especially user messages
	// that become adjacent after stripping an assistant approval prompt).
	var mergedOut []map[string]json.RawMessage
	for _, msg := range out {
		if len(mergedOut) == 0 {
			mergedOut = append(mergedOut, msg)
			continue
		}
		prev := mergedOut[len(mergedOut)-1]
		prevRole := rawMessageString(prev["role"])
		currRole := rawMessageString(msg["role"])
		if prevRole == currRole && currRole == "user" && canMergeAnthropicContent(prev["content"], msg["content"]) {
			mergedContent, err := mergeAnthropicContent(prev["content"], msg["content"])
			if err != nil {
				return SyntheticApprovalHistoryStripResult{Body: body}, err
			}
			prev["content"] = mergedContent
		} else {
			mergedOut = append(mergedOut, msg)
		}
	}
	out = mergedOut

	encoded, err := json.Marshal(out)
	if err != nil {
		return SyntheticApprovalHistoryStripResult{Body: body}, err
	}
	raw["messages"] = json.RawMessage(encoded)
	next, err := json.Marshal(raw)
	if err != nil {
		return SyntheticApprovalHistoryStripResult{Body: body}, err
	}
	return SyntheticApprovalHistoryStripResult{Body: next, Modified: true}, nil
}

func canMergeAnthropicContent(c1, c2 json.RawMessage) bool {
	var s1, s2 string
	err1 := json.Unmarshal(c1, &s1)
	err2 := json.Unmarshal(c2, &s2)
	if err1 == nil || err2 == nil {
		return err1 == nil && err2 == nil
	}
	var blocks1, blocks2 []json.RawMessage
	return json.Unmarshal(c1, &blocks1) == nil && json.Unmarshal(c2, &blocks2) == nil
}

func mergeAnthropicContent(c1, c2 json.RawMessage) (json.RawMessage, error) {
	if len(c1) == 0 {
		return c2, nil
	}
	if len(c2) == 0 {
		return c1, nil
	}

	var s1, s2 string
	err1 := json.Unmarshal(c1, &s1)
	err2 := json.Unmarshal(c2, &s2)

	if err1 == nil && err2 == nil {
		merged := s1 + "\n\n" + s2
		return json.Marshal(merged)
	}
	if err1 != nil && err2 == nil {
		var blocks1 []json.RawMessage
		if err := json.Unmarshal(c1, &blocks1); err != nil {
			return nil, err
		}
		blocks1 = append(blocks1, anthropicTextBlockRaw(s2))
		return json.Marshal(blocks1)
	}
	if err1 == nil && err2 != nil {
		var blocks2 []json.RawMessage
		if err := json.Unmarshal(c2, &blocks2); err != nil {
			return nil, err
		}
		out := make([]json.RawMessage, 0, len(blocks2)+1)
		out = append(out, anthropicTextBlockRaw(s1))
		out = append(out, blocks2...)
		return json.Marshal(out)
	}

	var blocks1 []json.RawMessage
	if err := json.Unmarshal(c1, &blocks1); err != nil {
		return nil, err
	}

	var blocks2 []json.RawMessage
	if err := json.Unmarshal(c2, &blocks2); err != nil {
		return nil, err
	}

	mergedBlocks := append(blocks1, blocks2...)
	return json.Marshal(mergedBlocks)
}

func anthropicTextBlockRaw(text string) json.RawMessage {
	block, _ := json.Marshal(map[string]string{
		"type": "text",
		"text": text,
	})
	return json.RawMessage(block)
}

func rawMessageString(raw json.RawMessage) string {
	var s string
	_ = json.Unmarshal(raw, &s)
	return s
}

func isSyntheticApprovalPromptText(text string) bool {
	return strings.Contains(text, InlineApprovalSubstitutedPromptMarker) ||
		strings.Contains(text, ToolApprovalSubstitutedPromptMarker)
}

func isBareSyntheticApprovalReply(text string) bool {
	if containsInlineApprovalAugmentationMarker(text) ||
		strings.Contains(text, InlineTaskDenyMarker) ||
		strings.Contains(text, InlineTaskCreatorErrorMarker) {
		return false
	}
	verb, _ := conversation.ParseApprovalReplyText(text)
	return verb != ""
}
