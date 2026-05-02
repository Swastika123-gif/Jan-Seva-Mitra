import streamlit as st
import pandas as pd
import pickle
import re
from pathlib import Path
import nltk
from nltk.corpus import stopwords

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(
    page_title="Jan Seva Mitra",
    page_icon="🇮🇳",
    layout="wide"
)

# ----------------------------
# NLTK Setup
# ----------------------------
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# ----------------------------
# CSS Styling
# ----------------------------
st.markdown("""
<style>

.main {
    background: linear-gradient(to right, #f7fbf4, #eef7ea);
    color: #1f2937 !important;
}

/* Force text visible */
html, body, [class*="css"] {
    color: #1f2937 !important;
}

h1, h2, h3, h4, h5, h6, p, li, span, div {
    color: #1f2937;
}

/* Hero */
.hero-box {
    background: linear-gradient(135deg, #0d47a1, #2e7d32);
    padding: 30px;
    border-radius: 20px;
    color: white !important;
    text-align: center;
    margin-bottom: 20px;
}

.hero-box h1,
.hero-box p {
    color: white !important;
}

/* Cards */
.info-card {
    background: #ffffff !important;
    color: #2f3e46 !important;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    border-left: 5px solid #2e7d32;
    margin-bottom: 15px;
}

.info-card h3 {
    color: #1b5e20 !important;
}

.info-card p,
.info-card li {
    color: #2f3e46 !important;
}

/* Result */
.result-card {
    background: #ffffff !important;
    color: #2f3e46 !important;
    padding: 18px;
    border-radius: 15px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
    margin-bottom: 15px;
}

.result-card h3 {
    color: #1b5e20 !important;
}

.result-card p {
    color: #2f3e46 !important;
}

/* Scheme */
.scheme-card {
    background: #ffffff !important;
    color: #2f3e46 !important;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #c8e6c9;
    margin-bottom: 10px;
}

.scheme-card h4 {
    color: #1b5e20 !important;
}

.scheme-card p {
    color: #2f3e46 !important;
}

/* Footer */
.footer-box {
    background: #eef7ea !important;
    color: #2f3e46 !important;
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #d7e9cf;
    margin-top: 18px;
}

.footer-box h4 {
    color: #1b5e20 !important;
}

.footer-box p {
    color: #2f3e46 !important;
}

/* Buttons */
.stButton>button {
    border-radius: 10px;
    height: 45px;
    font-weight: bold;
    background-color: #2e7d32;
    color: white !important;
}

.stButton>button:hover {
    background-color: #1b5e20;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)
# ----------------------------
# Load Model and Data
# ----------------------------
@st.cache_resource
def load_model():
    with open("models/svm_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


@st.cache_data
def load_schemes():
    return pd.read_csv("scheme_dataset.csv")


try:
    svm_model, vectorizer = load_model()
    scheme_df = load_schemes()
except Exception as e:
    st.error("Required model/data files are missing. Please check models folder and scheme_dataset.csv.")
    st.stop()

# ----------------------------
# Hindi Dictionaries
# ----------------------------
category_hindi = {
    "Water Supply": "जल आपूर्ति",
    "Electricity": "बिजली",
    "Roads": "सड़क",
    "Sanitation": "स्वच्छता",
    "Healthcare": "स्वास्थ्य सेवा",
    "Education": "शिक्षा",
    "Agriculture": "कृषि",
    "Public Safety": "सार्वजनिक सुरक्षा"
}

department_hindi = {
    "Water Department": "जल विभाग",
    "Electricity Board": "बिजली विभाग",
    "Public Works Department": "लोक निर्माण विभाग",
    "Sanitation Department": "स्वच्छता विभाग",
    "Health Department": "स्वास्थ्य विभाग",
    "Education Department": "शिक्षा विभाग",
    "Agriculture Department": "कृषि विभाग",
    "Police Department": "पुलिस विभाग",
    "Local Administration": "स्थानीय प्रशासन"
}

welfare_hindi = {
    "Public Health": "जन स्वास्थ्य",
    "Essential Services": "आवश्यक सेवाएँ",
    "Infrastructure": "बुनियादी ढांचा",
    "Cleanliness": "स्वच्छता सहायता",
    "Education Welfare": "शैक्षिक सहायता",
    "Farmer Support": "किसान सहायता",
    "Safety": "सुरक्षा सहायता",
    "General Welfare": "सामान्य कल्याण"
}

# ----------------------------
# Helper Functions
# ----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


def predict_category(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    return svm_model.predict(vec)[0]


def get_department(category):
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
    return department_map.get(category, "Local Administration")


def get_welfare_area(category):
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
    return welfare_map.get(category, "General Welfare")


def predict_priority(category, days_pending, affected_group):
    high_categories = {"Healthcare", "Public Safety", "Water Supply"}

    if category in high_categories and days_pending >= 2:
        return "High"
    if affected_group in {"Patients", "Women", "Students"} and days_pending >= 3:
        return "High"
    if days_pending >= 6:
        return "High"
    if days_pending >= 3:
        return "Medium"
    return "Low"


def recommend_schemes(predicted_category, citizen_type, income_group, location_type, age_group, gender):
    welfare_area = get_welfare_area(predicted_category)

    filtered = scheme_df[
        (scheme_df["welfare_area"] == welfare_area) &
        ((scheme_df["citizen_type"] == citizen_type) | (scheme_df["citizen_type"] == "General")) &
        ((scheme_df["income_group"] == income_group) | (scheme_df["income_group"] == "Low")) &
        ((scheme_df["location_type"] == location_type) | (scheme_df["location_type"] == "Both")) &
        ((scheme_df["age_group"] == age_group) | (scheme_df["age_group"] == "All")) &
        ((scheme_df["gender"] == gender) | (scheme_df["gender"] == "Any"))
    ]

    return welfare_area, filtered


def get_solution_details(category):
    solutions = {
        "Water Supply": {
            "problem": "This issue is related to water supply and may affect drinking water, household needs, hygiene, and public health.",
            "steps": [
                "Report the issue to the local Panchayat or ward office.",
                "Contact the Water Department for inspection.",
                "Mention the exact location and number of days the issue has continued."
            ],
            "gov_action": "Pipeline inspection, hand pump repair, water tank refill, or supply restoration may be required."
        },
        "Electricity": {
            "problem": "This issue is related to electricity supply, streetlights, voltage, transformer, or power reliability.",
            "steps": [
                "Inform the local electricity office or electricity board.",
                "Mention if the problem affects students, homes, shops, or farming work.",
                "Report how long the issue has continued and whether it happens daily."
            ],
            "gov_action": "Repair of transformer, electricity line, streetlight, or local power supply restoration may be needed."
        },
        "Roads": {
            "problem": "This complaint is related to damaged roads, unsafe travel, potholes, or poor public connectivity.",
            "steps": [
                "Report the issue to the Public Works Department or local authority.",
                "Mention if school children, patients, farmers, or vehicles are affected.",
                "Describe whether the road becomes dangerous during rain or night."
            ],
            "gov_action": "Road inspection, pothole repair, construction work, or maintenance action may be taken."
        },
        "Sanitation": {
            "problem": "This problem is related to cleanliness, garbage, drainage, sewage, or public hygiene.",
            "steps": [
                "Inform the sanitation department or municipal office.",
                "Mention if the issue is causing smell, disease risk, or blocked drainage.",
                "Report if the problem is near houses, schools, hospitals, or public places."
            ],
            "gov_action": "Cleaning work, garbage removal, drainage repair, or sanitation action may be taken."
        },
        "Healthcare": {
            "problem": "This issue is related to medical facilities, doctors, medicines, ambulance, or health service access.",
            "steps": [
                "Contact the nearest health center or district health office.",
                "Mention if patients are suffering due to delay or lack of service.",
                "Clearly state whether doctor, medicine, ambulance, or equipment is missing."
            ],
            "gov_action": "Health staff support, medicine supply, ambulance improvement, or medical service action may be required."
        },
        "Education": {
            "problem": "This issue is related to school facilities, teachers, classrooms, books, or student support.",
            "steps": [
                "Inform the school authority or education department.",
                "Mention whether students are unable to study properly.",
                "Highlight if building, teaching, books, toilets, or meals are inadequate."
            ],
            "gov_action": "School inspection, teaching improvement, infrastructure repair, or student support may be provided."
        },
        "Agriculture": {
            "problem": "This issue is related to farming support such as irrigation, fertilizer, seeds, crop loss, or farmer assistance.",
            "steps": [
                "Contact the agriculture officer or nearest Krishi Kendra.",
                "Mention whether irrigation, fertilizer, seeds, pests, or crop loss is the main problem.",
                "Keep details of crop damage, land area, or farming difficulty if needed."
            ],
            "gov_action": "Support may include irrigation help, agricultural schemes, subsidy, crop guidance, or field-level assistance."
        },
        "Public Safety": {
            "problem": "This issue is related to public safety, security, crime, harassment, or police support.",
            "steps": [
                "Contact the nearest police station or safety helpline.",
                "Mention whether women, children, elderly people, or families are affected.",
                "If the matter is serious, report urgently and keep records if possible."
            ],
            "gov_action": "Police action, patrolling, local safety support, or security measures may be initiated."
        }
    }

    return solutions.get(category, {
        "problem": "A general public issue has been identified.",
        "steps": ["Please contact the relevant local authority for further support."],
        "gov_action": "Appropriate action may be taken by the concerned department."
    })

# ----------------------------
# Session State
# ----------------------------
if "page" not in st.session_state:
    st.session_state.page = "awareness"

if "result_data" not in st.session_state:
    st.session_state.result_data = {}

# ----------------------------
# Sidebar Navigation
# ----------------------------
# ---------------- SIDEBAR ----------------
st.sidebar.markdown("""
<h2 style='color:#1b5e20;'>🇮🇳 Jan Seva Mitra</h2>
<p style='font-size:14px;'>Public Complaint & Guidance System</p>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Navigation buttons
if st.sidebar.button("📢 Know Your Rights"):
    st.session_state.page = "awareness"
    st.rerun()

if st.sidebar.button("📊 Seva Dashboard"):
    st.session_state.page = "home"
    st.rerun()

if st.sidebar.button("📝 Analyze Complaint"):
    st.session_state.page = "analysis"
    st.rerun()

st.sidebar.markdown("---")

# Info box
st.sidebar.markdown("""
<div style="background-color:#e8f5e9;padding:12px;border-radius:10px;">
<b>📌 Note</b><br>
This system provides guidance and awareness.<br>
Final resolution depends on official departments.
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<hr>
<p style='font-size:12px; color:gray;'>
Made for public awareness 🇮🇳
</p>
""", unsafe_allow_html=True)
# ----------------------------
# AWARENESS PAGE
# ----------------------------
if st.session_state.page == "awareness":

    banner_path = Path("assets/awareness_banner.png")
    if banner_path.exists():
        st.image(str(banner_path), use_container_width=True)

    st.markdown("""
    <div class="hero-box">
        <h1>Know Your Rights</h1>
        <p>Awareness & Life Guidance System</p>
        <p style="font-size:16px;">
        अपने अधिकार जानें – जीवन को बेहतर बनाने का मार्गदर्शन
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        <h3>📢 Why This Page Matters</h3>
        <p>
        Many people are unaware of their basic rights, education opportunities, and government support.
        This page helps you understand how to improve your life step by step.
        </p>
        <p>
        बहुत से लोग अपने अधिकारों, शिक्षा के अवसरों और सरकारी सहायता के बारे में नहीं जानते।
        यह पेज आपको बेहतर जीवन के लिए मार्गदर्शन देता है।
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h3>🌱 Understanding Poverty / गरीबी को समझें</h3>
        <p>
        Poverty is not only about lack of money. It also includes lack of education, healthcare,
        skills, awareness, and opportunities. A better life starts with awareness and correct decisions.
        </p>
        <p>
        गरीबी केवल पैसे की कमी नहीं है। यह शिक्षा, स्वास्थ्य, कौशल, जानकारी और अवसरों की कमी से भी जुड़ी होती है।
        बेहतर जीवन की शुरुआत जागरूकता और सही निर्णय से होती है।
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="info-card">
            <h3>🎓 Education is the strongest path</h3>
            <ul>
                <li>Send children to school regularly.</li>
                <li>Do not make children leave school for work.</li>
                <li>Government schools, scholarships, and support schemes can help poor families.</li>
            </ul>
            <p>
            शिक्षा बच्चों का भविष्य बदल सकती है। बच्चों को काम पर नहीं, स्कूल भेजें।
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-card">
            <h3>🏛 Use Government Support</h3>
            <ul>
                <li>Free or low-cost education in government schools.</li>
                <li>Scholarships for SC/ST and low-income families.</li>
                <li>Mid-day meal and student support schemes.</li>
                <li>Government hospitals and health schemes.</li>
            </ul>
            <p>
            सरकारी सुविधाओं की जानकारी लें और सही जगह आवेदन करें।
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h3>💡 Steps to Improve Life / जीवन सुधार के उपाय</h3>
        <ul>
            <li>Educate children and avoid child labour.</li>
            <li>Develop skills for better income.</li>
            <li>Avoid unnecessary loans and use loans carefully.</li>
            <li>Maintain cleanliness, health, and proper documents.</li>
            <li>Ask for information from school, Panchayat, government office, or online portals.</li>
        </ul>
        <p>
        बच्चों की पढ़ाई जारी रखें, अनावश्यक कर्ज से बचें, सरकारी योजनाओं की जानकारी लें और कौशल सीखें।
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer-box">
        <h4>📢 Message for Everyone / सभी के लिए संदेश</h4>
        <p>
        A better future is possible with education, awareness, health, skills, and right decisions.
        Small steps today can change tomorrow.
        </p>
        <p>
        शिक्षा, जागरूकता, स्वास्थ्य, कौशल और सही निर्णय से बेहतर भविष्य संभव है।
        आज के छोटे कदम कल जीवन बदल सकते हैं।
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h3>💬 Inspiration</h3>
        <p>
        "Education and awareness are the strongest tools to break poverty and build a better future."
        </p>
        <p>
        "शिक्षा और जागरूकता ही गरीबी को दूर करने का सबसे मजबूत साधन है।"
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Continue to Home / आगे बढ़ें", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()

# ----------------------------
# HOME PAGE
# ----------------------------
elif st.session_state.page == "home":

    banner_path = Path("assets/banner.png")
    if banner_path.exists():
        st.image(str(banner_path), use_container_width=True)

    st.markdown("""
    <div class="hero-box">
        <h1>Seva Dashboard</h1>
        <p>Public Service Overview</p>
        <p>जन सेवा सहायता डैशबोर्ड</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h3>Welcome / स्वागत है</h3>
        <p>
        Jan Seva Mitra helps citizens understand public problems clearly.
        It analyzes complaints, identifies the responsible department, estimates urgency,
        and suggests useful welfare direction and public support.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h3>📢 Important Information / महत्वपूर्ण जानकारी</h3>
        <p>
        This system provides guidance to help you understand your problem clearly.
        It helps identify the issue, suggests possible solutions, and guides you toward the next steps.
        However, users are advised to contact the concerned department or local authority for final resolution.
        </p>
        <p>
        यह प्रणाली आपकी समस्या को समझने में सहायता करती है।
        यह समस्या की पहचान करने, संभावित समाधान सुझाने और आगे की कार्रवाई के लिए मार्गदर्शन प्रदान करती है।
        अंतिम समाधान के लिए संबंधित विभाग या स्थानीय प्रशासन से संपर्क करना आवश्यक है।
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="info-card">
            <h3>📢 Complaint Analysis</h3>
            <p>Understand issues related to water, electricity, sanitation, roads, safety, education, healthcare, and agriculture.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-card">
            <h3>🏛 Department Suggestion</h3>
            <p>The system helps identify the department that may be responsible for the reported issue.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="info-card">
            <h3>🎯 Public Support Guidance</h3>
            <p>It suggests welfare direction and relevant schemes based on complaint and citizen profile.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h3>Who can use this platform? / यह मंच किनके लिए उपयोगी है?</h3>
        <ul>
            <li>Farmers facing agriculture or irrigation issues</li>
            <li>Students facing education or school problems</li>
            <li>Families facing water, sanitation, or electricity issues</li>
            <li>Citizens seeking public complaint guidance</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    chart_path = Path("assets/model_comparison.png")
    if chart_path.exists():
        st.markdown('<div class="section-title">Model Performance Dashboard</div>', unsafe_allow_html=True)
        st.image(str(chart_path), caption="Model Comparison on Complaint Dataset", use_container_width=True)

    if st.button("Start Complaint Analysis / शिकायत विश्लेषण शुरू करें", use_container_width=True):
        st.session_state.page = "analysis"
        st.rerun()

# ----------------------------
# ANALYSIS PAGE
# ----------------------------
elif st.session_state.page == "analysis":
    st.title("📝 Complaint Analysis Form")

    st.markdown("""
    <div class="info-card">
        <h3>🧾 How to Write Complaint / शिकायत कैसे लिखें?</h3>
        <ul>
            <li>Write your problem clearly in simple words.</li>
            <li>Mention issue, place, and how long it has continued.</li>
            <li>Clear complaint text gives better guidance.</li>
        </ul>
        <p><b>Examples / उदाहरण:</b></p>
        <ul>
            <li>There is no drinking water in our village for many days.</li>
            <li>Farmers are facing irrigation problems in the area.</li>
            <li>Electricity supply is irregular and students cannot study at night.</li>
            <li>Road near the school is damaged and unsafe.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    complaint = st.text_area(
        "Enter Complaint / अपनी शिकायत लिखें",
        placeholder="Example: There is a serious issue of no irrigation facilities in our village for many days",
        height=160
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        citizen_type = st.selectbox("Citizen Type / नागरिक प्रकार", ["General", "Farmer", "Student", "Woman", "Patient", "Senior Citizen"])
        income_group = st.selectbox("Income Group / आय वर्ग", ["Low", "Medium"])

    with col2:
        location_type = st.selectbox("Location Type / क्षेत्र", ["Rural", "Urban"])
        age_group = st.selectbox("Age Group / आयु वर्ग", ["Child", "Adult", "Senior", "All"])

    with col3:
        gender = st.selectbox("Gender / लिंग", ["Any", "Male", "Female"])
        days_pending = st.slider("Days Pending / कितने दिनों से समस्या है", 1, 15, 3)

    affected_group = st.selectbox(
        "Affected Group / प्रभावित समूह",
        ["Residents", "Families", "Farmers", "Students", "Patients", "Women"]
    )

    a1, a2 = st.columns(2)

    with a1:
        if st.button("Analyze Now / अभी विश्लेषण करें", use_container_width=True):
            if complaint.strip() == "":
                st.warning("Please enter a complaint first.")
            else:
                with st.spinner("🔍 Analyzing complaint... Please wait"):
                    predicted_category = predict_category(complaint)
                    department = get_department(predicted_category)
                    welfare_area = get_welfare_area(predicted_category)
                    priority = predict_priority(predicted_category, days_pending, affected_group)

                _, recommended = recommend_schemes(
                    predicted_category,
                    citizen_type,
                    income_group,
                    location_type,
                    age_group,
                    gender
                )

                st.session_state.result_data = {
                    "complaint": complaint,
                    "predicted_category": predicted_category,
                    "department": department,
                    "welfare_area": welfare_area,
                    "priority": priority,
                    "recommended": recommended
                }

                st.session_state.page = "result"
                st.rerun()

    with a2:
        if st.button("Back to Home / होम पेज पर वापस जाएँ", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()

# ----------------------------
# RESULT PAGE
# ----------------------------
elif st.session_state.page == "result":
    data = st.session_state.result_data
    solution = get_solution_details(data["predicted_category"])

    st.title("📊 Complaint Analysis Result")

    st.markdown("""
    <div class="info-card">
        <h3>Result Summary / परिणाम सारांश</h3>
        <p>
        Your complaint has been analyzed successfully. Below are category, priority,
        department, welfare area, solution guidance, and suggested schemes.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-card">
        <h3>Complaint Entered / दर्ज शिकायत</h3>
        <p>{data['complaint']}</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"""
        <div class="result-card">
            <h3>Predicted Category / अनुमानित श्रेणी</h3>
            <p>{data['predicted_category']} / {category_hindi.get(data['predicted_category'], '')}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-card">
            <h3>Department / विभाग</h3>
            <p>{data['department']} / {department_hindi.get(data['department'], '')}</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="result-card">
            <h3>Welfare Area / सहायता क्षेत्र</h3>
            <p>{data['welfare_area']} / {welfare_hindi.get(data['welfare_area'], '')}</p>
        </div>
        """, unsafe_allow_html=True)

        if data["priority"] == "High":
            st.error("🔴 High Priority / उच्च प्राथमिकता")
        elif data["priority"] == "Medium":
            st.warning("🟠 Medium Priority / मध्यम प्राथमिकता")
        else:
            st.success("🟢 Low Priority / कम प्राथमिकता")

    st.markdown("""
    <div class="info-card">
        <h3>📌 Problem Understanding / समस्या की जानकारी</h3>
    </div>
    """, unsafe_allow_html=True)
    st.write(solution.get("problem", "General issue identified."))

    st.markdown("""
    <div class="info-card">
        <h3>🛠 What You Should Do / आपको क्या करना चाहिए?</h3>
    </div>
    """, unsafe_allow_html=True)

    for step in solution.get("steps", []):
        st.write("👉", step)

    st.markdown("""
    <div class="info-card">
        <h3>🏛 Expected Government Action / संभावित सरकारी कार्रवाई</h3>
    </div>
    """, unsafe_allow_html=True)
    st.write(solution.get("gov_action", "Action will be taken by the concerned department."))

    st.subheader("🎯 Suggested Schemes / सुझाई गई योजनाएँ")

    recommended = data["recommended"]

    if len(recommended) > 0:
        for _, row in recommended.iterrows():
            st.markdown(f"""
            <div class="scheme-card">
                <h4>{row['scheme_name']}</h4>
                <p class="small-note">
                    Welfare Area: <b>{row['welfare_area']}</b><br>
                    Citizen Type: <b>{row['citizen_type']}</b><br>
                    Location: <b>{row['location_type']}</b><br>
                    Age Group: <b>{row['age_group']}</b>
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("No matching schemes found for the selected details.")

    st.markdown("""
    <div class="footer-box">
        <h4>📌 Final Guidance / अंतिम मार्गदर्शन</h4>
        <p>
        This system provides guidance to help you understand your problem clearly.
        It helps identify the issue, suggests possible solutions, and guides you toward the next steps.
        However, users are advised to contact the concerned department or local authority for final resolution.
        </p>
        <p>
        यह प्रणाली आपकी समस्या को समझने में सहायता करती है।
        यह समस्या की पहचान करने, संभावित समाधान सुझाने और आगे की कार्रवाई के लिए मार्गदर्शन प्रदान करती है।
        अंतिम समाधान के लिए संबंधित विभाग या स्थानीय प्रशासन से संपर्क करना आवश्यक है।
        </p>
    </div>
    """, unsafe_allow_html=True)

    r1, r2 = st.columns(2)

    with r1:
        if st.button("Analyze Another Complaint / दूसरी शिकायत का विश्लेषण करें", use_container_width=True):
            st.session_state.page = "analysis"
            st.rerun()

    with r2:
        if st.button("Go to Home Page / होम पेज पर जाएँ", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()