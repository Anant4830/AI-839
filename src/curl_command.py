import numpy as np
import json

# Generate random data
checking_status = np.random.choice(['<0', '0<=X<200', '>=200', 'no checking'], 1)[0]
duration = np.random.randint(1, 100, 1)[0]

# Ensure duration is a standard Python int
duration = int(duration)

# Prepare the curl command
data = {
    "columns": ["checking_status", "duration"],
    "data": [[checking_status, duration]]
}

# Convert to JSON string for the curl command
json_data = json.dumps(data)

# Construct the curl command
curl_command = f'curl -X POST -H "Content-Type: application/json" -d \'{json_data}\' http://127.0.0.1:5000/invocations'

# Print the curl command
print(curl_command)

# Optionally, run the curl command (uncomment the following line to execute)
import subprocess
subprocess.run(curl_command, shell=True)
