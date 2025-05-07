import os
import shutil
from googleapiclient.http import MediaFileUpload
from googleapiclient.discovery import build
from google.oauth2 import service_account
import gdown

SCOPES = ["https://www.googleapis.com/auth/drive"]
SERVICE_ACCOUNT_FILE = "service_account.json"
PARENT_FOLDER_ID = "18gAv2YaI5ai_mKW-6xyOxC7bpnRw3cHN"


def remove_ckpt_files(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".ckpt"):
                file_path = os.path.join(root, file)
                os.remove(file_path)


def authenticate():
    creds = service_account.Credentials.from_service_account_file(
        os.path.join("src", "saver", SERVICE_ACCOUNT_FILE), scopes=SCOPES
    )
    return creds


def download_auth_file():
    if not os.path.exists(os.path.join("src", "saver", SERVICE_ACCOUNT_FILE)):
        gdown.download(
            id="1jOWRnYFbsjtNQczeMSqviWr1nxAZIQqi", output=SERVICE_ACCOUNT_FILE
        )


def folder_exists(service, folder_name, parent_folder_id):
    query = f"'{parent_folder_id}' in parents and name = '{folder_name}' and trashed = false"
    results = service.files().list(q=query, fields="files(id)").execute()
    items = results.get("files", [])
    return items[0]["id"] if items else None


def upload_file(service, file_path, parent_folder_id):
    file_metadata = {"name": os.path.basename(file_path), "parents": [parent_folder_id]}
    media = MediaFileUpload(file_path, mimetype="application/zip")
    file = (
        service.files()
        .create(body=file_metadata, media_body=media, fields="id")
        .execute()
    )
    print(f"Uploaded file {os.path.basename(file_path)} with ID: {file.get('id')}")


def zip_and_upload(folder):
    folder_to_upload = "/".join(folder.split("/")[1:]).replace("/", "_")
    zip_name = f"{folder_to_upload}.zip"
    parent_dir = folder.split("/")[0]
    base_name = folder.split("/")[1]
    shutil.make_archive(zip_name[:-4], "zip", parent_dir, base_name)
    return zip_name


def upload_folder(folder):
    download_auth_file()
    creds = authenticate()
    service = build("drive", "v3", credentials=creds)
    remove_ckpt_files(folder)
    zip_name = zip_and_upload(folder)
    upload_file(service, zip_name, PARENT_FOLDER_ID)
