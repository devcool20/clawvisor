ALTER TABLE tasks ADD COLUMN max_cost_micros INTEGER;
ALTER TABLE tasks ADD COLUMN max_tokens INTEGER;

ALTER TABLE agent_runtime_settings ADD COLUMN max_cost_micros INTEGER;
ALTER TABLE agent_runtime_settings ADD COLUMN max_tokens INTEGER;
