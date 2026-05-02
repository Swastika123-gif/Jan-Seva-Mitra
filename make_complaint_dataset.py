import pandas as pd
import random

category_map = {
    "Water Supply": [
        "no drinking water",
        "hand pump not working",
        "pipeline damaged",
        "water tank not filled",
        "no tap water available"
    ],
    "Electricity": [
        "power cut",
        "transformer damaged",
        "voltage fluctuation",
        "streetlight not working",
        "no electricity at night"
    ],
    "Roads": [
        "road broken near school",
        "potholes on road",
        "road not constructed properly",
        "bridge damaged",
        "road needs repair"
    ],
    "Sanitation": [
        "garbage not cleaned",
        "blocked drain causing smell",
        "no dustbin in village",
        "waste collection not regular",
        "dirty surroundings near houses"
    ],
    "Healthcare": [
        "no doctor available",
        "medicines not available",
        "long waiting time in hospital",
        "health center not functioning",
        "ambulance delayed"
    ],
    "Education": [
        "teacher absent in school",
        "no proper classrooms",
        "lack of study materials",
        "school building needs repair",
        "poor teaching quality"
    ],
    "Agriculture": [
        "no irrigation facilities",
        "crop damage due to pests",
        "fertilizer not available",
        "lack of farming equipment",
        "crop insurance claim delayed"
    ],
    "Public Safety": [
        "no police patrolling",
        "theft cases increasing",
        "unsafe roads at night",
        "harassment complaints ignored",
        "crime rate increasing"
    ]
}

department_map = {
    "Water Supply": "Water Department",
    "Electricity": "Electricity Board",
    "Roads": "Public Works Department",
    "Sanitation": "Sanitation Department",
    "Healthcare": "Health Department",
    "Education": "Education Department",
    "Agriculture": "Agriculture Department",
    "Public Safety": "Police Department"
}

welfare_map = {
    "Water Supply": "Public Health",
    "Electricity": "Essential Services",
    "Roads": "Infrastructure",
    "Sanitation": "Cleanliness",
    "Healthcare": "Public Health",
    "Education": "Education Welfare",
    "Agriculture": "Farmer Support",
    "Public Safety": "Safety"
}

priority_map = {
    "Water Supply": ["High", "Medium"],
    "Electricity": ["High", "Medium", "Low"],
    "Roads": ["High", "Medium"],
    "Sanitation": ["High", "Medium", "Low"],
    "Healthcare": ["High", "Medium"],
    "Education": ["High", "Medium"],
    "Agriculture": ["High", "Medium"],
    "Public Safety": ["High", "Medium"]
}

affected_map = {
    "Water Supply": ["Families", "Residents"],
    "Electricity": ["Residents", "Students"],
    "Roads": ["Residents", "Students"],
    "Sanitation": ["Residents", "Families"],
    "Healthcare": ["Patients"],
    "Education": ["Students"],
    "Agriculture": ["Farmers"],
    "Public Safety": ["Residents", "Women"]
}

location_options = ["Rural", "Urban"]

prefixes = [
    "There is a serious issue of",
    "People are facing problem due to",
    "Complaint has been reported about",
    "Residents are suffering because of",
    "Urgent attention is needed for"
]

suffixes = [
    "in our village",
    "in the local area",
    "for many days",
    "since last week",
    "in this locality"
]

data = []

for category, texts in category_map.items():
    for _ in range(25):   # 25 rows per category = 200 rows
        base_text = random.choice(texts)

        templates = [
            f"{random.choice(prefixes)} {base_text} {random.choice(suffixes)}",
            f"{base_text} is creating problems {random.choice(suffixes)}",
            f"People reported that {base_text} {random.choice(suffixes)}",
            f"{base_text} has become a serious concern {random.choice(suffixes)}",
            f"Local residents said that {base_text} {random.choice(suffixes)}"
        ]

        complaint = random.choice(templates)
        priority = random.choice(priority_map[category])
        department = department_map[category]
        location = random.choice(location_options)
        days_pending = random.randint(1, 10)
        affected = random.choice(affected_map[category])
        welfare = welfare_map[category]

        data.append([
            complaint,
            category,
            priority,
            department,
            location,
            days_pending,
            affected,
            welfare
        ])

df = pd.DataFrame(data, columns=[
    "complaint_text",
    "category",
    "priority",
    "department",
    "location_type",
    "days_pending",
    "affected_group",
    "welfare_area"
])

df.to_csv("complaint_dataset.csv", index=False)

print("complaint_dataset.csv created successfully")
print(df.head())
print("\nDataset shape:", df.shape)