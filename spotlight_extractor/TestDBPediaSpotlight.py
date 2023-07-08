from datetime import time

import requests
from requests.auth import HTTPBasicAuth

text = "President Obama called Wednesday on Congress to extend a tax break for students included in last year's economic stimulus package, arguing that the policy provides more generous assistance."

parameters={}
parameters['text']=text
parameters['confidence']=0.35

Headers = {"Accept" :"application/json"}

api_endpoint = "https://api.dbpedia-spotlight.org/en/annotate"

response = requests.get(api_endpoint,params=parameters,headers=Headers)

print(response.headers)

#print(response.headers["X-RateLimitRemaining"])
#print(response.headers["X-RateLimit-Retry-After"])
#print(response.headers["X-RateLimit-Limit"])
if response.status_code ==200:
    print(response.json(), response.elapsed.total_seconds())
elif response.status_code == 429:
    time.sleep(int(response.headers["Retry-After"]))
else:
    print(f"Error code {response.status_code}, {response.content}")


