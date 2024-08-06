import csv
import requests

# read csv file
with open('./example.csv',encoding='utf-8') as f:
  reader = csv.DictReader(f)
  points = [row for row in reader]
  


# results
results = []

for point in points:
  ID1 = point['czcID']
  ID2 = point['OBJECTID'] 
  lon1 = point['lon1']
  lat1 = point['lat1']  
  lon2 = point['lon2']
  lat2 = point['lat2']

  
  full_url = 'https://api.map.baidu.com/routematrix/v2/walking?output=json&origins='+lat1+','+lon1+'&destinations='+lat2+','+lon2+'&ak=your own key'


  # requests
  response = requests.get(full_url)
  data = response.json()
  print(data)
  if 'status' in data and data['status'] != 0:
    print("Fail:", data['message'])
    continue
  

  # get results of requests
  result = data['result'][0]
  distance = result['distance']['text'] 
  duration = result['duration']['text']

  # add results
  results.append({
    'ID1': ID1,
    'ID2': ID2,
    'lon1': lon1,
    'lat1': lat1,
    'lon2': lon2, 
    'lat2': lat2,
    'distance': distance,
    'duration': duration
  })

# 保存结果
with open('./output.csv', 'w') as f:
  writer = csv.DictWriter(f, fieldnames=results[0].keys())
  writer.writeheader()
  writer.writerows(results)