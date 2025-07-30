#!/usr/bin/env python3

import requests
import sys

def check_health():
    try:
        response = requests.get('http://localhost:8000/')
        if response.status_code == 200:
            print('Health check passed!')
            return 0
        else:
            print(f'Health check failed with status code: {response.status_code}')
            return 1
    except Exception as e:
        print(f'Health check failed with error: {str(e)}')
        return 1

if __name__ == '__main__':
    sys.exit(check_health())