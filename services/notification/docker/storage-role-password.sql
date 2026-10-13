-- supabase/postgres creates supabase_storage_admin without a password, so storage-api can't authenticate

-- This image logs statements, including on error — without these the password lands in the
-- container log in plaintext.
SET log_statement = 'none';
SET pgaudit.log = 'none';
SET log_min_error_statement = 'panic';

\getenv pgpass POSTGRES_PASSWORD
ALTER USER supabase_storage_admin WITH PASSWORD :'pgpass';
