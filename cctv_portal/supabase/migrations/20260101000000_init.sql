-- Create violations table
CREATE TABLE IF NOT EXISTS violations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    camera_id TEXT,
    candidate_id TEXT,
    violation_type TEXT,
    description TEXT,
    severity TEXT,
    evidence_url TEXT,
    bbox JSONB,
    metadata JSONB
);

-- Enable RLS
ALTER TABLE violations ENABLE ROW LEVEL SECURITY;

-- Policy: Everyone can read violations (for dashboard demo)
CREATE POLICY "Public Read Violations" ON violations
    FOR SELECT USING (true);

-- Policy: Service role can insert (for backend)
CREATE POLICY "Service Role Insert Violations" ON violations
    FOR INSERT WITH CHECK (true);

-- Create storage bucket for evidence if not exists
-- Note: This usually needs to be done via the Supabase Dashboard or API, 
-- but we define it here for reference.
-- INSERT INTO storage.buckets (id, name, public) VALUES ('evidence', 'evidence', true);
