import os
import datetime
import pytz
import pickle

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from googleapiclient.discovery import build

# Define the scopes for the API
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']


def map_timezone(timezone_str):
    """Map user-provided timezone to valid pytz timezone name."""
    timezone_str = timezone_str.strip()
    if timezone_str == 'UTC':
        return 'UTC'
    elif timezone_str.startswith('UTC+'):
        offset = timezone_str.replace('UTC+', '')
        return f'Etc/GMT-{offset}'  # Note the minus sign
    elif timezone_str.startswith('UTC-'):
        offset = timezone_str.replace('UTC-', '')
        return f'Etc/GMT+{offset}'  # Note the plus sign
    else:
        # You can add more mappings or validations here
        return timezone_str  # Assume it's a valid pytz timezone

def get_calendar_service():
    """Authenticate and return the Google Calendar service."""
    creds = None
    # Token file stores user's access and refresh tokens
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If credentials are invalid or don't exist, start the OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # The file 'credentials.json' should be in your working directory
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for future runs
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    # Build the service
    service = build('calendar', 'v3', credentials=creds)
    return service

def get_events_for_date(date, timezone='UTC'):
    """
    Get events for a specific date.

    Args:
        date (datetime.date): The date for which to retrieve events.
        timezone (str): Timezone string, e.g., 'America/Los_Angeles'.

    Returns:
        List of events.
    """
    service = get_calendar_service()
    timezone_mapped = map_timezone(timezone)
    # Set timezone
    tz = pytz.timezone(timezone_mapped)

    # Start and end of the day in ISO format
    start_of_day = tz.localize(datetime.datetime.combine(date, datetime.time.min)).isoformat()
    end_of_day = tz.localize(datetime.datetime.combine(date, datetime.time.max)).isoformat()

    # Fetch events
    events_result = service.events().list(calendarId='primary', timeMin=start_of_day,
                                          timeMax=end_of_day, singleEvents=True,
                                          orderBy='startTime').execute()
    events = events_result.get('items', [])
    return events

def get_events_for_today(timezone='UTC'):
    """
    Get events for today.

    Args:
        timezone (str): Timezone string, e.g., 'America/Los_Angeles'.

    Returns:
        List of events.
    """
    today = datetime.date.today()
    return get_events_for_date(today, timezone)

def get_events_for_date_str(date_str: str, timezone: str = 'UTC') -> str:
    """
    Get events for a specific date in the specified timezone.

    Args:
        date_str (str): The date in 'YYYY-MM-DD' format.
        timezone (str): Timezone string, e.g., 'America/Los_Angeles'.

    Returns:
        str: A string containing the list of events.
    """
    import datetime
    date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
    events = get_events_for_date(date, timezone)
    if not events:
        return f'No events found for {date_str} in timezone {timezone}.'
    else:
        event_list = [
            f"{event['start'].get('dateTime', event['start'].get('date'))}: {event.get('summary', 'No Title')}"
            for event in events
        ]
        return '\n'.join(event_list)


def get_events_for_today_str(timezone: str = 'UTC') -> str:
    """
    Get events for today in the specified timezone.

    Args:
        timezone (str): Timezone string, e.g., 'America/Los_Angeles'.

    Returns:
        str: A string containing the list of events.
    """
    events = get_events_for_today(timezone)
    if not events:
        return f'No events found for today in timezone {timezone}.'
    else:
        event_list = [
            f"{event['start'].get('dateTime', event['start'].get('date'))}: {event.get('summary', 'No Title')}"
            for event in events
        ]
        return '\n'.join(event_list)

