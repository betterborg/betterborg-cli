-- Schema version 2. Immutable repository analysis and generated prompt history.

CREATE TABLE repository_analyses (
    id TEXT PRIMARY KEY,
    repository_id TEXT NOT NULL REFERENCES repositories(id) ON DELETE RESTRICT,
    head_sha TEXT NOT NULL,
    summary TEXT NOT NULL,
    primary_language TEXT NOT NULL,
    is_monorepo INTEGER NOT NULL CHECK (is_monorepo IN (0, 1)),
    overall_score REAL NOT NULL CHECK (overall_score BETWEEN 0 AND 5),
    analysis_json TEXT NOT NULL,
    prior_analysis_id TEXT,
    score_delta REAL,
    created_at TEXT NOT NULL,
    UNIQUE (id, repository_id),
    FOREIGN KEY (prior_analysis_id, repository_id)
        REFERENCES repository_analyses(id, repository_id) ON DELETE RESTRICT,
    CHECK ((prior_analysis_id IS NULL) = (score_delta IS NULL))
);

CREATE INDEX idx_repository_analyses_repository_created
    ON repository_analyses(repository_id, created_at, id);

CREATE TABLE repository_packages (
    id TEXT PRIMARY KEY,
    repository_id TEXT NOT NULL,
    analysis_id TEXT NOT NULL,
    package_path TEXT NOT NULL,
    package_name TEXT NOT NULL,
    primary_language TEXT NOT NULL,
    rubric_json TEXT NOT NULL,
    overall_score REAL NOT NULL CHECK (overall_score BETWEEN 0 AND 5),
    UNIQUE (analysis_id, package_path),
    FOREIGN KEY (analysis_id, repository_id)
        REFERENCES repository_analyses(id, repository_id) ON DELETE RESTRICT
);

CREATE INDEX idx_repository_packages_analysis_path
    ON repository_packages(analysis_id, package_path);

CREATE TABLE generated_prompts (
    id TEXT PRIMARY KEY,
    repository_id TEXT NOT NULL,
    analysis_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('coding', 'review', 'merge')),
    version INTEGER NOT NULL CHECK (version > 0),
    body_md TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    UNIQUE (repository_id, role, version),
    FOREIGN KEY (analysis_id, repository_id)
        REFERENCES repository_analyses(id, repository_id) ON DELETE RESTRICT
);

CREATE INDEX idx_generated_prompts_latest
    ON generated_prompts(repository_id, role, version DESC);

CREATE TRIGGER repository_analyses_are_append_only_on_update
BEFORE UPDATE ON repository_analyses
BEGIN
    SELECT RAISE(ABORT, 'repository analyses are append-only');
END;

CREATE TRIGGER repository_analyses_are_append_only_on_delete
BEFORE DELETE ON repository_analyses
BEGIN
    SELECT RAISE(ABORT, 'repository analyses are append-only');
END;

CREATE TRIGGER repository_packages_are_append_only_on_update
BEFORE UPDATE ON repository_packages
BEGIN
    SELECT RAISE(ABORT, 'repository packages are append-only');
END;

CREATE TRIGGER repository_packages_are_append_only_on_delete
BEFORE DELETE ON repository_packages
BEGIN
    SELECT RAISE(ABORT, 'repository packages are append-only');
END;

CREATE TRIGGER generated_prompts_are_append_only_on_update
BEFORE UPDATE ON generated_prompts
BEGIN
    SELECT RAISE(ABORT, 'generated prompts are append-only');
END;

CREATE TRIGGER generated_prompts_are_append_only_on_delete
BEFORE DELETE ON generated_prompts
BEGIN
    SELECT RAISE(ABORT, 'generated prompts are append-only');
END;
