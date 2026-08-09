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


-- Phase C: saved & shareable dashboards ------------------------------------
create table if not exists public.dashboards (
  id           uuid primary key default gen_random_uuid(),
  owner_id     text not null,
  owner_email  text,
  name         text not null,
  config       jsonb not null default '{}',   -- {charts:[...]} etc.
  columns      jsonb not null default '[]',   -- data snapshot: headers
  rows         jsonb not null default '[]',   -- data snapshot: values
  is_public    boolean not null default false,
  share_token  uuid not null default gen_random_uuid(),
  created_at   timestamptz not null default now()
);
create index if not exists dashboards_owner_idx on public.dashboards (owner_id);
create index if not exists dashboards_token_idx on public.dashboards (share_token);
alter table public.dashboards enable row level security;

create table if not exists public.dashboard_shares (
  id                uuid primary key default gen_random_uuid(),
  dashboard_id      uuid not null references public.dashboards(id) on delete cascade,
  shared_with_email text not null,
  created_at        timestamptz not null default now()
);
create index if not exists dashboard_shares_email_idx on public.dashboard_shares (shared_with_email);
alter table public.dashboard_shares enable row level security;
