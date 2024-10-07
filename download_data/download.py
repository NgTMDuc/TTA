from __future__ import print_function
import os
import io
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload

# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive']

def authenticate():
    """Authenticate the user using OAuth 2.0 and get the Google Drive service object."""
    creds = None
    # Check if token.json already exists (it stores the user's access and refresh tokens)
    if os.path.exists('token2.json'):
        creds = Credentials.from_authorized_user_file('token2.json', SCOPES)
    # If there are no valid credentials, ask the user to authenticate.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secret_3.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token2.json', 'w') as token:
            token.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)

def download_file(service, file_id, destination):
    """Download a file from Google Drive by file ID and save it to the destination path."""
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(destination, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}%.")
    print(f"File downloaded to: {destination}")

def main():
    """Main function to authenticate and download a file from Google Drive."""
    # Authenticate and get the Google Drive service
    service = authenticate()

    # Specify the file ID and destination path for download
    file_id = '15aiZpiQpQzYwWWSSpwKHf7wKo65j-oF4'  # Replace with the actual file ID from Google Drive
    destination = '/mnt/ducntm/DATA/blur.tar'  # Replace with the desired file path and name

    # Call the download function
    download_file(service, file_id, destination)
    
    # file_id = "15vLMParMqQDpDe34qXTq1eAwZCK4OU_K"
    # destination = "aiotlab/Data/digital.tar"
    # download_file(service, file_id, destination)
    
    # file_id = "1LjYf2LMhSPfSdCYR9DFZj2N24ix84fds"
    # destination = "aiotlab/Data/extra.tar"
    # download_file(service, file_id, destination)
    
    # file_id = "1w05DJwhGz66zXTA0WK1ie9R54-ZmCtGB"
    # destination = "aiotlab/Data/noise.tar"
    # download_file(service, file_id, destination)
    
    # file_id = "1IGdjgLrQocafIIYLs_r_skfOq24oNbB6"
    # destination = "aiotlab/Data/weather.tar"
    # download_file(service, file_id, destination)
if __name__ == '__main__':
    main()
