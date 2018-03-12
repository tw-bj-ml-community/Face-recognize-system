-- run "sqlite3 dev.db < initial_database.sql" on "fr_system" folder to create a brand new dev db

CREATE TABLE user_info (id INTEGER PRIMARY KEY, user_name TEXT NOT NULL);
CREATE TABLE face_feature (id INTEGER PRIMARY KEY,user_id INTEGER , face_feature BLOB,FOREIGN KEY(user_id) REFERENCES user_info(id));

