from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from passlib.context import CryptContext
from sympy.integrals.meijerint_doc import category
from endpoints.database import get_db, engine
from endpoints import models
from endpoints.models import Users, Documents, Chats
from typing import List, Dict
from sqlalchemy import func
import boto3
from io import BytesIO
from uuid import uuid4
from datetime import datetime
from ocr.run import process_file
from rag.embeddings import handle_chat_embeddings
from rag.llm import query_llm
from rag.category import classify_document_content
from rag.qdrant_utils import client, collection_exists
from rag.chatname import create_chat_name
import os


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
models.Base.metadata.create_all(bind=engine)
S3_BUCKET = os.getenv("S3_BUCKET")
S3_REGION = os.getenv("S3_REGION")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")

s3 = boto3.client(
    's3',
    region_name=S3_REGION,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY
)




MIME_TYPE_MAP = {
    'application/pdf': 'pdf',
    'application/msword': 'doc',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
    'text/plain': 'txt',
    'image/jpeg': 'jpg',
    'image/png': 'png',
}
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Pydantic models
class UserCreate(BaseModel):
    email: str
    password: str
    user_type: str

class UserLogin(BaseModel):
    email: str
    password: str
    user_type: str

class FolderCountResponse(BaseModel):
    foldername: str
    count: int
    timestamp: str

class DocumentResponse(BaseModel):
    document_url: str
    timestamp: str  # The timestamp will now be returned as a string

class ChatResponse(BaseModel):
    chat_name: str
    latest_timestamp: str

class ImportantDocumentResponse(BaseModel):
    document_url: str
    timestamp: str

class DocumentByCategoryResponse(BaseModel):
    document_url: str
    timestamp: str

class FileUploadQueryParams(BaseModel):
    user_id: int

class MarkImportantRequest(BaseModel):
    user_id: int
    doc_url: str

class MoveToTrashRequest(BaseModel):
    user_id: int
    doc_url: str

class DocumentQueryRequest(BaseModel):
    user_id: int
    category: str

class FolderQueryRequest(BaseModel):
    user_id: int
    foldername: str

# Hash password function
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

# Verify password function
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# Signup endpoint
@app.post("/signup", response_model=dict)
async def signup(user: UserCreate, db: Session = Depends(get_db)):
    # Check if user exists
    existing_user = db.query(Users).filter(Users.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User already exists")

    # Hash the user's password before storing
    hashed_password = hash_password(user.password)
    new_user = Users(email=user.email, password=hashed_password, user_type=user.user_type)

    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User created successfully"}


@app.post("/login", response_model=dict)
async def login(user: UserLogin, db: Session = Depends(get_db)):
    # Fetch the user by email and user_type
    db_user = db.query(Users).filter(
        Users.email == user.email,
        Users.user_type == user.user_type
    ).first()

    # Check if user exists and password is correct
    if not db_user or not verify_password(user.password, db_user.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials or user type")

    # Return the email and user_id on successful login
    return {
        "message": "Login successful",
        "email": db_user.email,
        "user_id": db_user.user_id
    }


@app.post("/upload_files")
async def upload_files(
    query_params: FileUploadQueryParams = Depends(),  # Dependency to extract query parameters
    files: List[UploadFile] = File(...),  # List of files in the request body
    db: Session = Depends(get_db)
):
    uploaded_files = []
    user_id = query_params.user_id  # Access user_id from query parameters

    for file in files:
        # Generate unique file ID and key
        file_id = str(uuid4())
        file_key = f"documents/{file_id}_{file.filename}"

        # Get content type and set document type
        content_type = file.content_type
        doc_type = MIME_TYPE_MAP.get(content_type)
        if not doc_type:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported file type")

        # Read file content into memory
        file_content = await file.read()

        # Upload the file to S3
        try:
            s3.upload_fileobj(BytesIO(file_content), S3_BUCKET, file_key)

            # Set Content-Disposition and ContentType metadata
            s3.copy_object(
                Bucket=S3_BUCKET,
                CopySource={'Bucket': S3_BUCKET, 'Key': file_key},
                Key=file_key,
                MetadataDirective='REPLACE',
                ContentDisposition='inline',
                ContentType=content_type
            )

        except ClientError as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="File upload failed")

        # Create a public URL for the uploaded file
        doc_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{file_key}"

        # Insert file metadata into the database
        new_document = Documents(
            user_id=user_id,
            category=None,
            is_important=False,
            is_deleted=False,
            document_url=doc_url,
            chat_name=None,
            doctype=doc_type,
            foldername=None,
            timestamp=datetime.utcnow()
        )

        db.add(new_document)
        db.commit()
        db.refresh(new_document)

        # Append uploaded file info to the response list
        uploaded_files.append({"document_url": doc_url, "doctype": doc_type})

    return {"message": "Files uploaded successfully", "uploaded_files": uploaded_files}


@app.post("/upload-folder")
async def upload_folder(user_id: int, foldername: str, files: List[UploadFile] = File(...),
                        db: Session = Depends(get_db)):
    uploaded_files = []

    for file in files:
        # Generate unique file ID and key within the folder
        file_id = str(uuid4())
        file_key = f"{foldername}/{file_id}_{file.filename}"  # Including foldername in S3 key

        # Get content type and set document type
        content_type = file.content_type
        doc_type = MIME_TYPE_MAP.get(content_type)
        if not doc_type:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported file type")

        # Read file content into memory
        file_content = await file.read()

        # Upload the file to S3
        try:
            s3.upload_fileobj(BytesIO(file_content), S3_BUCKET, file_key)

            # Set Content-Disposition and ContentType metadata
            s3.copy_object(
                Bucket=S3_BUCKET,
                CopySource={'Bucket': S3_BUCKET, 'Key': file_key},
                Key=file_key,
                MetadataDirective='REPLACE',
                ContentDisposition='inline',
                ContentType=content_type
            )

        except ClientError as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="File upload failed")

        # Create a public URL for the uploaded file
        doc_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{file_key}"

        # Insert file metadata into the database
        new_document = Documents(
            user_id=user_id,
            category=None,
            is_important=False,
            is_deleted=False,
            document_url=doc_url,
            chat_name=None,
            doctype=doc_type,
            foldername=foldername,
            timestamp=datetime.utcnow()
        )

        db.add(new_document)
        db.commit()
        db.refresh(new_document)

        # Append uploaded file info to the response list
        uploaded_files.append({"document_url": doc_url, "doctype": doc_type})

    return {"message": "Folder uploaded successfully", "foldername": foldername, "uploaded_files": uploaded_files}


@app.get("/user/{user_id}/folders", response_model=List[FolderCountResponse])
async def get_user_folders(user_id: int, db: Session = Depends(get_db)):
    # Query all folder names and their timestamps for the given user_id
    foldernames_with_timestamp = db.query(Documents.foldername, Documents.timestamp).filter(
        Documents.user_id == user_id
    ).all()

    if not foldernames_with_timestamp:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No folders found for this user")

    # Create a dictionary to count the occurrences of each folder name, along with the latest timestamp
    folder_count_dict = {}

    for folder, timestamp in foldernames_with_timestamp:
        if folder:
            if folder in folder_count_dict:
                folder_count_dict[folder]["count"] += 1
            else:
                folder_count_dict[folder] = {"count": 1, "timestamp": timestamp}

    # Prepare the response by converting the dictionary into a list of FolderCountResponse
    folder_counts = [
        {
            "foldername": foldername,
            "count": data["count"],
            "timestamp": data["timestamp"].strftime("%B %d, %Y")  # Format the timestamp
        }
        for foldername, data in folder_count_dict.items()
    ]

    return folder_counts



@app.get("/user/{user_id}/documents", response_model=List[DocumentResponse])
async def get_documents_by_timestamp(user_id: int, db: Session = Depends(get_db)):
    # Get the current timestamp
    current_timestamp = datetime.utcnow()

    # Query the documents table for document URLs and timestamps, with the specified user_id and timestamp condition
    documents = db.query(Documents.document_url, Documents.timestamp).filter(
        Documents.user_id == user_id,
        Documents.timestamp <= current_timestamp
    ).order_by(Documents.timestamp.desc()).all()  # Sort by timestamp in descending order (most recent first)

    # Check if documents were found
    if not documents:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="No documents found for this user at or after the current timestamp.")

    # Prepare the response by converting the result into a list of DocumentResponse objects
    document_responses = [
        {
            "document_url": doc[0],
            "timestamp": doc[1].strftime("%B %d, %Y")  # Explicitly format the datetime to string
        } for doc in documents
    ]

    return document_responses

@app.get("/search-documents/", response_model=List[DocumentResponse])
async def search_documents(name: str, user_id: int, db: Session = Depends(get_db)):
    # Query to get all documents for the specified user
    results = db.query(Documents.document_url, Documents.timestamp).filter(
        Documents.user_id == user_id
    ).all()

    # Filter documents based on the extracted name from `doc_url`
    matching_docs = []
    for doc_url, timestamp in results:
        full_filename = doc_url.split('/')[-1]
        actual_filename = full_filename.split('_')[-1].split('.')[0]

        if actual_filename.lower() == name.lower():
            # Format the timestamp
            formatted_timestamp = timestamp.strftime("%B %d, %Y")
            matching_docs.append(DocumentResponse(document_url=doc_url, timestamp=formatted_timestamp))

    if not matching_docs:
        raise HTTPException(status_code=404, detail="No documents found with the specified name")

    return matching_docs

@app.get("/user/{user_id}/prev_chats", response_model=List[ChatResponse])
async def get_user_chats(user_id: int, db: Session = Depends(get_db)):
    # Query all unique chat names and the latest timestamp for each chat for the specified user
    results = (
        db.query(Chats.chat_name, func.min(Chats.timestamp).label("latest_timestamp"))
        .filter(Chats.user_id == user_id)
        .group_by(Chats.chat_name)
        .order_by(func.min(Chats.timestamp).desc())
        .all()
    )

    if not results:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No chats found for this user")

    # Prepare the response by formatting each timestamp
    chat_list = []
    for chat_name, latest_timestamp in results:
        # Format the timestamp to "Month Day, Year" format
        formatted_timestamp = latest_timestamp.strftime("%B %d, %Y")
        chat_list.append(ChatResponse(chat_name=chat_name, latest_timestamp=formatted_timestamp))

    return chat_list

@app.put("/documents/mark-important/")
async def mark_document_as_important(request: MarkImportantRequest, db: Session = Depends(get_db)):
    # Find the document with the specified user_id and doc_url
    document = db.query(Documents).filter(
        Documents.user_id == request.user_id,
        Documents.document_url == request.doc_url
    ).first()

    # Check if the document exists
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    # Update the is_important field to True
    document.is_important = True
    db.commit()

    return {"message": "Document marked as important successfully"}

@app.put("/documents/move-trash/")
async def move_trash(request: MoveToTrashRequest, db: Session = Depends(get_db)):
    # Find the document with the specified user_id and doc_url
    document = db.query(Documents).filter(
        Documents.user_id == request.user_id,
        Documents.document_url == request.doc_url
    ).first()

    # Check if the document exists
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    # Update the is_deleted field to True
    document.is_deleted = True
    db.commit()

    return {"message": "Document moved to trash"}

@app.get("/user/{user_id}/category/{category_name}/documents", response_model=List[DocumentByCategoryResponse])
async def get_documents_by_category(user_id: int, category_name: str, db: Session = Depends(get_db)):
    # Query the documents for the specified user_id and category_name
    documents = db.query(Documents).filter(
        Documents.user_id == user_id,
        Documents.category_name == category_name
    ).all()

    # Check if any documents are found for the category
    if not documents:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No documents found for this category and user")

    # Format the response with document URL and formatted timestamp
    response = [
        {
            "document_url": doc.document_url,
            "timestamp": doc.timestamp.strftime("%B %d, %Y")  # Format to "Month Day, Year"
        }
        for doc in documents
    ]

    return response

@app.get("/user/{user_id}/important-documents", response_model=List[ImportantDocumentResponse])
async def get_important_documents(user_id: int, db: Session = Depends(get_db)):
    # Query the documents where is_important is True for the given user_id
    important_docs = db.query(Documents).filter(
        Documents.user_id == user_id,
        Documents.is_important == True
    ).all()

    # Check if any important documents are found
    if not important_docs:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No important documents found for this user")

    # Format the response with document URL and formatted timestamp
    response = [
        {
            "document_url": doc.document_url,
            "timestamp": doc.timestamp.strftime("%B %d, %Y")  # Format to "Month Day, Year"
        }
        for doc in important_docs
    ]

    return response

@app.get("/user/{user_id}/trash-documents", response_model=List[ImportantDocumentResponse])
async def get_trash_documents(user_id: int, db: Session = Depends(get_db)):
    # Query the documents where is_important is True for the given user_id
    important_docs = db.query(Documents).filter(
        Documents.user_id == user_id,
        Documents.is_deleted == True
    ).all()

    # Check if any important documents are found
    if not important_docs:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No important documents found for this user")

    # Format the response with document URL and formatted timestamp
    response = [
        {
            "document_url": doc.document_url,
            "timestamp": doc.timestamp.strftime("%B %d, %Y")  # Format to "Month Day, Year"
        }
        for doc in important_docs
    ]

    return response

@app.post("/documents/search-by-category/")
async def get_documents_by_category(request: DocumentQueryRequest, db: Session = Depends(get_db)):
    # Query the documents table for documents that match the user_id and category
    documents = db.query(Documents).filter(
        Documents.user_id == request.user_id,
        Documents.category == request.category
    ).all()

    if not documents:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No documents found for this category")

    # Prepare the response with formatted timestamps
    document_data = [
        {
            "document_url": doc.document_url,
            "timestamp": doc.timestamp.strftime("%B %d, %Y")
        }
        for doc in documents
    ]

    return document_data

@app.post("/documents/search-by-folder/")
async def get_documents_by_folder(request: FolderQueryRequest, db: Session = Depends(get_db)):
    # Query the documents table for documents that match the user_id and foldername
    documents = db.query(Documents).filter(
        Documents.user_id == request.user_id,
        Documents.foldername == request.foldername
    ).all()

    if not documents:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No documents found for this folder")

    # Prepare the response with formatted timestamps
    document_data = [
        {
            "document_url": doc.document_url,
            "timestamp": doc.timestamp.strftime("%B %d, %Y")
        }
        for doc in documents
    ]

    return document_data

@app.post("/upload_and_initialize/")
async def upload_and_initialize(file: UploadFile = File(...), user_id: int = Form(...),
    query: str = Form(...),db: Session = Depends(get_db)):
    content_type = file.content_type
    file_id = str(uuid4())
    file_key = f"documents/{file_id}_{file.filename}"
    doc_type = MIME_TYPE_MAP.get(content_type)
    if not doc_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported file type"
        )
    file_content = await file.read()
    try:
        s3.upload_fileobj(BytesIO(file_content), S3_BUCKET, file_key)
        s3.copy_object(
                    Bucket=S3_BUCKET,
                    CopySource={"Bucket": S3_BUCKET, "Key": file_key},
                    Key=file_key,
                    MetadataDirective="REPLACE",
                    ContentDisposition="inline",
                    ContentType=content_type,
                )
    except ClientError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,detail="File upload failed",)


    doc_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{file_key}"
    document_text = process_file(file_content, content_type)
    category=classify_document_content(document_text)
    print(category)
    chat_name = create_chat_name(document_text, query)
    print(chat_name)
    if not collection_exists(chat_name):
        handle_chat_embeddings(chat_name, document_text)

    response = query_llm(chat_name, query)

    
    new_document = Documents(
        user_id=user_id,
        category=category,
        is_important=False,
        is_deleted=False,
        document_url=doc_url,
        chat_name=chat_name,
        doctype=doc_type,
        foldername=None,
        timestamp=datetime.utcnow(),
    )

    db.add(new_document)
    db.commit()
    db.refresh(new_document)

    new_chat = Chats(
        chat_name=chat_name,
        query=query,
        response=response,
        timestamp=datetime.utcnow(),
        user_id=user_id
    )
    db.add(new_chat)
    db.commit()
    db.refresh(new_chat)
    return JSONResponse(content={"chat_name": chat_name, "initial_response": response})



@app.post("/chat/")
async def chat(chat_name: str = Form(...), user_id: int = Form(...), query: str = Form(...), db: Session = Depends(get_db)):
    print(f"Received chat_name: {chat_name}, user_id: {user_id}, query: {query}")

    # Validate incoming data
    if not chat_name or not query:
        raise HTTPException(status_code=400, detail="chat_name and query are required.")
    if not isinstance(user_id, int):
        raise HTTPException(status_code=400, detail="user_id must be an integer.")

    if not collection_exists(chat_name):
        raise HTTPException(status_code=404, detail="Chat not initialized or chat name not found.")

    response = query_llm(chat_name, query)
    new_chat = Chats(
        chat_name=chat_name,
        query=query,
        response=response,
        timestamp=datetime.utcnow(),
        user_id=user_id
    )
    db.add(new_chat)
    db.commit()
    db.refresh(new_chat)
    return JSONResponse(content={"response": response['result']})


@app.post("/upload_file_to_collection/")
async def upload_files_to_chat(
    chat_name: str = Form(...),
    user_id: int = Form(...),
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    if not collection_exists(chat_name):
        raise HTTPException(status_code=404, detail="Chat collection not found.")

    uploaded_files = []
    errors = []

    for file in files:
        content_type = file.content_type
        file_id = str(uuid4())
        file_key = f"documents/{file_id}_{file.filename}"
        doc_type = MIME_TYPE_MAP.get(content_type)
        content = await file.read()

        if not doc_type:
            errors.append({"filename": file.filename, "error": "Unsupported file type"})
            continue

        try:
            # Upload to S3
            s3.upload_fileobj(BytesIO(content), S3_BUCKET, file_key)
            s3.copy_object(
                Bucket=S3_BUCKET,
                CopySource={"Bucket": S3_BUCKET, "Key": file_key},
                Key=file_key,
                MetadataDirective="REPLACE",
                ContentDisposition="inline",
                ContentType=content_type,
            )

            # Generate file URL
            doc_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{file_key}"

            # Process file and extract text
            document_text = process_file(content, content_type)

            # Classify content and handle embeddings
            category = classify_document_content(document_text)
            handle_chat_embeddings(chat_name, document_text)

            # Save metadata to database
            new_document = Documents(
                user_id=user_id,
                category=category,
                is_important=False,
                is_deleted=False,
                document_url=doc_url,
                chat_name=chat_name,
                doctype=doc_type,
                foldername=None,
                timestamp=datetime.utcnow(),
            )

            db.add(new_document)
            db.commit()
            db.refresh(new_document)

            uploaded_files.append({
                "filename": file.filename,
                "url": doc_url,
                "message": "Uploaded successfully",
            })

        except ClientError as e:
            errors.append({
                "filename": file.filename,
                "error": "File upload failed due to server error",
            })

    return JSONResponse(content={
        "uploaded_files": uploaded_files,
        "errors": errors,
    })


# Optionally, an endpoint to delete a chat collection
@app.delete("/delete_chat/{chat_name}")
def delete_chat(chat_name: str):
    client.delete_collection(collection_name=chat_name)
    return {"message": f"Chat collection '{chat_name}' deleted successfully."}

@app.get("/get_chats_by_chatnames/")
async def get_chats_by_chatname(user_id: int, chat_name: str, db: Session = Depends(get_db)):
    try:
        # Query the chats table based on chat_name and order by timestamp
        chats = db.query(Chats).filter(
            Chats.user_id == user_id, Chats.chat_name == chat_name
        ).order_by(Chats.timestamp.asc()).all()

        # Query the documents table based on chat_name and order by uploaded_at
        documents = db.query(Documents).filter(
            Documents.user_id == user_id, Documents.chat_name == chat_name
        ).order_by(Documents.timestamp.asc()).all()

        # Check if there are any chats or documents with the given chat_name
        if not chats and not documents:
            raise HTTPException(
                status_code=404, detail=f"No chats or documents found for chat name: {chat_name}"
            )

        # Prepare the response data by combining chats and documents
        combined_data = []

        # Add document items with a type marker and `uploaded_at` timestamp
        for doc in documents:
            combined_data.append({
                "type": "document",
                "url": doc.document_url,
                "file_type": doc.doctype,
                "timestamp": doc.timestamp  # Use uploaded_at as timestamp for documents
            })

        # Add chat items with a type marker
        for chat in chats:
            combined_data.append({
                "type": "chat",
                "query": chat.query,
                "response": chat.response,
                "timestamp": chat.timestamp
            })

        # Sort the combined list by timestamp, prioritizing documents over chats when timestamps are identical
        combined_data = sorted(
            combined_data,
            key=lambda x: (x["timestamp"], 0 if x["type"] == "document" else 1)
        )

        return {"chat_name": chat_name, "items": combined_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


