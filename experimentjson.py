import json

with open('database.json', 'r') as file:
    data = json.load(file)

user_names = [user["name"] for user in data["users"]]
print(user_names)


import json

with open('database.json', 'r') as file:
    data = json.load(file)





print(json.dumps(data, indent=4))


area_ids = [
    area["area_id"]
    for user in data["users"]
    for area in user["saved_areas"]
]

print(area_ids)


area_info = next(
    (
        area
        for user in data["users"]
        for area in user["saved_areas"]
        if area["area_id"] == "FIELD-01"
    ),
    None
)

print(area_info)

