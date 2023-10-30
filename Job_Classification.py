import csv
import requests
import nltk
import joblib
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def connect_to_mongo_db():
    client = pymongo.MongoClient("mongodb+srv://nawodyaa59:zLnU8ZSjBaBqSyq5@cluster0.7c9abpa.mongodb.net/?retryWrites=true&w=majority")
    db = client["cv_data"]
    collection = db["cv_collection"]

# Load models
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

api_url = 'https://backend-brixc.koyeb.app/managecandidates/managecandidates'

def classifier_model(vectorizer, label_encoder):
    data = pd.read_csv("job_roles.csv")

    X = vectorizer.fit_transform(data["Job Role Description"])
    y = label_encoder.fit_transform(data["Required Skills"])

    # Train a Random Forest classifier
    classifier = RandomForestClassifier()
    classifier.fit(X, y)
    
    return classifier

st.title("Best Fit Candidates")

def get_unique():
    start_role = set()

    with open('job_roles.csv', mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            start_role.add(row[0])

    return sorted(list(start_role))

start_role = get_unique()
#start_role = {'IT Procurement Specialist', 'Big Data Engineer', 'IT Change Manager', 'CRM Administrator', 'SharePoint Administrator', 'IT Project Manager', 'IT Operations Manager', 'Infrastructure Engineer', 'Software Architect', 'Cloud Engineer', 'Software Development Manager', 'Back-End Developer', 'IT Trainer', 'Full Stack Developer', 'Salesforce Developer', 'WordPress Developer', 'Cybersecurity Analyst', 'DevOps Engineer', 'IT Compliance Manager', 'Graphic Designer', 'Cloud Security Engineer', 'Embedded Systems Engineer', 'SAP Consultant', 'Kotlin Developer', 'QA Engineer', 'Data Scientist', 'IT Auditor', 'UX Researcher', 'HR Manager', 'Natural Language Processing (NLP) Engineer', 'IT Service Delivery Manager', 'Systems Administrator', 'SharePoint Developer', 'SAP Basis Administrator', 'Flutter Developer', 'System Integration Engineer', 'Machine Learning Engineer', 'Ruby on Rails Developer', 'E-commerce Developer', 'ERP Developer', 'Project Manager', 'Marketing Specialist', 'Chatbot Developer', 'IT Business Development Manager', 'Network Engineer', 'Cloud Solutions Architect', 'Software Engineer', 'IT Support Specialist', 'CRM Developer', 'IT Asset Manager', 'Data Engineer', 'UI/UX Designer', 'Game Developer', 'Mobile App Developer', 'AI Engineer', 'Cloud Architect', 'Automation Engineer', 'Security Engineer', 'Robotic Process Automation (RPA) Developer', 'Compiler Engineer', 'Full Stack JavaScript Developer', 'Android Developer', 'iOS Developer', 'Unity Developer', 'IT Business Analyst', 'Blockchain Developer', 'IT Risk Manager', 'Business Intelligence Developer', 'AR/VR Developer', 'Front-End Developer', 'Site Reliability Engineer (SRE)', 'Backend API Engineer', 'Robotics Engineer', 'Business Systems Analyst', 'Quality Assurance Manager', 'IT Consultant', 'IT Sales Representative', 'Digital Marketing Analyst', 'IT Security Analyst'}

import webbrowser

with st.sidebar:

    if st.button("Back To Home"):
      #time.sleep(0.5) 
      webbrowser.open_new_tab("http://localhost:8501/")

    selected_role = st.sidebar.radio(
    "Select your Job Role",
    get_unique()
)

st.markdown(
    """<style>
div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
    font-size: 24px;
}
    </style>
    """, unsafe_allow_html=True)

#option = st.selectbox('Select Job Role', start_role)

predicted_job_role = []
st.subheader(f'Selected Job Role : {selected_role}')

if st.button('Get Skills'):
    
    # Vectorize user input
    user_description_vector = vectorizer.transform([selected_role]) 

    # Predict label 
    classifier = classifier_model(vectorizer, label_encoder)
    predicted_label = classifier.predict(user_description_vector)[0]

    # Decode label
    predicted_job_role = label_encoder.inverse_transform([predicted_label])[0]
    
    st.success(f"Predicted Skill Set: {predicted_job_role}")
    
st.title("Top Candidates")

skills = predicted_job_role

if skills:
    user_skills = [skill.strip() for skill in skills.split(',')]
    ps = PorterStemmer()
    user_skill_tokens = set()
    for user_skill in user_skills:
        user_skill_tokens.update(
            [ps.stem(word) for word in word_tokenize(user_skill) if word.lower() not in stopwords.words('english')])

    data = []
    with open('CvDatast.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)

    matching_individuals = []
    for person in data:
        person_skills = person['Technical_Skills'].split(', ')
        person_skill_tokens = set()
        for person_skill in person_skills:
           person_skill_tokens.update([ps.stem(word) for word in word_tokenize(person_skill) if  
                                        word.lower() not in stopwords.words('english')])
        matching_tokens = user_skill_tokens.intersection(person_skill_tokens)
        if matching_tokens:
            person['Matching_Tokens'] = matching_tokens
            matching_individuals.append(person)

    def score_individual(individual):
        education_weight = {'Bachelor\'s': 1,'Bsc\'s': 1, 'MSA\'s': 2,'MSc\'s': 2, 'Master\'s': 2, 'Doctoral': 3, 'Phd': 3}
        skill_weight = {'Python': 5, 'Java': 2, 'SQL': 3, 'Machine Learning': 4, 'Hadoop': 2, 'Spark': 3, 'Hive': 2, 'node js': 1, 'react': 2, 'spring boot': 2, 'git': 2, 'android studio': 3, 'tableau': 4, 'snowflake': 4}
        experience_weight = 0.5
        education_score = education_weight.get(individual['Edu_Qualifications'], 0)
        skill_score = skill_weight.get(individual['Technical_Skills'], 0) 
        experience_score = float(individual['Experience_years']) * experience_weight
        matching_tokens_score = len(individual.get('Matching_Tokens', []))
        total_score = education_score + skill_score + experience_score + matching_tokens_score
        individual['Total_Score'] = total_score
        return total_score

    sorted_individuals = sorted(matching_individuals, key=score_individual, reverse=True)

    top_10_individuals = sorted_individuals[:10]

    if top_10_individuals:
        
        results = []
        printed_candidates = set()

        for person in top_10_individuals:
            if person['Name'] not in printed_candidates:
                printed_candidates.add(person['Name'])
                results.append({"Rank": len(results) + 1, 
                "CVNumber": person['CVNumber'],              
                "Name": person['Name'],
                "Contact": person.get('Contact', ''),
                "Score": round(person['Total_Score'], 2)})

                data = {
                  "candidate_id": "5f4e50a52f72e06d3b18588d",
                  "candidate_external_id": person['CVNumber'],
                  "candidate_name": person['Name'],
                  "candidate_email": f"{person['Name']}@gmail.com",
                  "candidate_status": 10,
                  "candidate_contact_number": person.get('Contact', ''),
                  "job_id": "653ba340695250e6267edb88",
                  "meeting_id": "",
                  "join_url": "",
                  "option": "insert"
                }

                response = requests.post(api_url, json=data)

                if response.status_code == 200:
                    print("Request was successful. Response content:")
                    print(response.text)
                else:
                    print("Request failed with status code:", response.status_code)

        results_df = pd.DataFrame(results)
        
        st.table(results_df.style.format({'Score': '{:.2f}'}).set_precision(2))  

        csv = results_df.to_csv(index=False)

        st.download_button(
           label="Download",
           data=csv,
           file_name='Top_Candidates.csv',
           mime='text/csv',
        )
    else:
        st.write("No matching candidates found")
