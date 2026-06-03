ALTER TABLE tasks ADD COLUMN IF NOT EXISTS max_cost_micros BIGINT;
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS max_tokens BIGINT;

ALTER TABLE agent_runtime_settings ADD COLUMN IF NOT EXISTS max_cost_micros BIGINT;
ALTER TABLE agent_runtime_settings ADD COLUMN IF NOT EXISTS max_tokens BIGINT;
