import os
import pickle
from datetime import datetime, timedelta
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from typing import List, Dict

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

class EmailFetcher:
    def __init__(self, credentials_path: str = 'credentials.json'):
        self.credentials_path = credentials_path
        self.creds = None
        self.service = None
        self.authenticate()

    def authenticate(self):
        """Authenticate with Gmail API."""
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                self.creds = pickle.load(token)

        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, SCOPES)
                self.creds = flow.run_local_server(port=8080)
            
            with open('token.pickle', 'wb') as token:
                pickle.dump(self.creds, token)

        self.service = build('gmail', 'v1', credentials=self.creds)

    def fetch_recent_emails(self, days: int = 1) -> List[Dict]:
        """Fetch emails from the last specified days."""
        try:
            # Calculate the time threshold
            time_threshold = datetime.now() - timedelta(days=days)
            query = f'after:{time_threshold.strftime("%Y/%m/%d")}'

            results = self.service.users().messages().list(
                userId='me', q=query).execute()
            messages = results.get('messages', [])

            emails = []
            for message in messages:
                msg = self.service.users().messages().get(
                    userId='me', id=message['id'], format='full').execute()
                
                # Extract headers
                headers = msg['payload']['headers']
                subject = next(h['value'] for h in headers if h['name'] == 'Subject')
                sender = next(h['value'] for h in headers if h['name'] == 'From')
                date = next(h['value'] for h in headers if h['name'] == 'Date')

                # Extract body
                body = ''
                if 'parts' in msg['payload']:
                    for part in msg['payload']['parts']:
                        if part['mimeType'] == 'text/plain':
                            body = part['body'].get('data', '')
                elif 'body' in msg['payload']:
                    body = msg['payload']['body'].get('data', '')

                emails.append({
                    'id': message['id'],
                    'subject': subject,
                    'sender': sender,
                    'date': date,
                    'body': body,
                    'snippet': msg.get('snippet', '')
                })

            return emails
        except Exception as e:
            print(f"Error fetching emails: {str(e)}")
            return [] 