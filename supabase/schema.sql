-- Phase B: saved datasets
-- Run this once in Supabase → SQL Editor → New query → Run.

create table if not exists public.datasets (
  id          uuid primary key default gen_random_uuid(),
  user_id     text not null,               -- Supabase user id (from the JWT 'sub')
  user_email  text,
  name        text not null,
  columns     jsonb not null default '[]', -- column headers
  rows        jsonb not null default '[]', -- row values (array of arrays)
  row_count   integer,
  created_at  timestamptz not null default now()
);

create index if not exists datasets_user_id_idx on public.datasets (user_id);

-- Lock the table down. The backend talks to this table with the service_role
-- key (which bypasses RLS) and scopes every query by user_id itself. Enabling
-- RLS with NO policies means the public anon/publishable key cannot read or
-- write this table directly — only the trusted backend can.
alter table public.datasets enable row level security;
