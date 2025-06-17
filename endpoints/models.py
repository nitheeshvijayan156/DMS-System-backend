from sqlalchemy import Column, ForeignKey, String, Text, TIMESTAMP, Integer, Boolean, text
from sqlalchemy.orm import relationship
from endpoints.database import Base

class Users(Base):
    __tablename__ = "users"
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(100), nullable=False)
    password = Column(String(100), nullable=False)
    user_type = Column(String(100), nullable=False)

    documents = relationship("Documents", back_populates="user", cascade="all, delete-orphan")
    chats = relationship("Chats", back_populates="user", cascade="all, delete-orphan")

class Documents(Base):
    __tablename__ = "documents"
    doc_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    category = Column(String(100), nullable=True)
    is_important = Column(Boolean, default=False)
    is_deleted = Column(Boolean, default=False)
    document_url = Column(String(255), nullable=True)
    chat_name = Column(String(255), nullable=True)
    doctype = Column(String(255), nullable=False)
    foldername = Column(String(100), nullable=True)
    timestamp = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))

    user = relationship("Users", back_populates="documents")

class Chats(Base):
    __tablename__ = "chats"
    chat_id = Column(Integer, primary_key=True, autoincrement=True)
    chat_name = Column(String(255), nullable=False)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    timestamp = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)

    user = relationship("Users", back_populates="chats")
