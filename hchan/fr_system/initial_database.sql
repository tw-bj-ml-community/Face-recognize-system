-- run this script by using "sqlite3 auction.db < create.sql" to create a brand new dev db

CREATE TABLE user_info (id INTEGER PRIMARY KEY, user_name TEXT NOT NULL);
CREATE TABLE face_feature (id INTEGER PRIMARY KEY,user_id INTEGER , face_feature BLOB,FOREIGN KEY(user_id) REFERENCES user_info(id));

