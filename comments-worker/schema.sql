CREATE TABLE IF NOT EXISTS comments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  page_slug TEXT NOT NULL,
  parent_id INTEGER,
  nickname TEXT NOT NULL,
  content TEXT NOT NULL,
  is_author INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  FOREIGN KEY (parent_id) REFERENCES comments(id)
);

CREATE INDEX IF NOT EXISTS idx_comments_page_slug ON comments(page_slug);
CREATE INDEX IF NOT EXISTS idx_comments_parent_id ON comments(parent_id);
