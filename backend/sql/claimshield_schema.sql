create extension if not exists pgcrypto;
create table if not exists users (id uuid primary key default gen_random_uuid(), user_id text unique not null, name text, email text, created_at timestamptz default now());
create table if not exists claims (id uuid primary key default gen_random_uuid(), claim_id text unique not null, user_id text not null, claim_description text not null, claim_amount numeric, image_reference text, video_reference text, image_hash text, video_hash text, text_guard jsonb, image_guard jsonb, video_guard jsonb, temporal_analysis jsonb, media_authenticity jsonb, security_evidence jsonb, llm_analysis jsonb, final_route text, security_flag boolean, created_at timestamptz default now());
create index if not exists claims_user_id_idx on claims(user_id);
create index if not exists claims_created_at_idx on claims(created_at);
