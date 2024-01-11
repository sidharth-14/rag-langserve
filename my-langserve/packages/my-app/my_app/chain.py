from langchain.prompts import ChatPromptTemplate
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores.faiss import FAISS
import requests
import geocoder
from math import radians, sin, cos, sqrt, atan2

hf_api_token="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"
repo_id=  "TinyLlama/TinyLlama-1.1B-Chat-v1.0"       #"codellama/CodeLlama-7b-hf"

def get_current_location():
    location = geocoder.ip('me')
    latitude = location.latlng[0]
    longitude = location.latlng[1]
    return latitude, longitude

def places_distance(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert coordinates from degrees to radians
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    # Calculate the differences between latitudes and longitudes
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Calculate the distance
    distance = R * c

    return distance

def get_elevation_data(latitude, longitude):
    api_url = "http://115.112.141.157:8585/elevation_query"
    input_data = {
        "latitude": latitude,
        "longitude": longitude
    }
    response = requests.post(api_url, json=input_data)
    data = response.json()
    # Extract elevation from the results
    elevation = None
    if "results" in data and data["results"]:
        elevation = data["results"][0].get("elevation")
        return elevation
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None
    
def get_places_data():
    current_latitude, current_longitude = get_current_location()
    api_url = "http://115.112.141.157:8686/overpass_query"
    input_data = {
        "radius": 1000,
        "latitude": current_latitude,
        "longitude": current_longitude,
        "tags": [
                    {
                        "name": "amenity",
                        "value": ""
                    }
                ]
            }
    # Make the API call
    response = requests.post(api_url, json=input_data)
    
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()

        # Extract and store latitude, longitude, and amenity name
        data_list = []
        for element in data.get("elements", []):
            if element["type"] == "node" and "lat" in element and "lon" in element and "tags" in element:
                latitude = element["lat"]
                longitude = element["lon"]
                amenity = element["tags"].get("amenity", "")
                name = element["tags"].get("name", "")
                
                elevation = get_elevation_data(latitude,longitude)

                distance = places_distance(current_latitude, current_longitude, latitude, longitude)

                data_list.append(f'''"name": "{name}", "amenity":"{amenity}", "latitude": "{latitude}", "longitude": "{longitude}", "elevation": "{elevation}", "distance": {distance:.6f}''')

        # Sort the list based on distance
        data_list.sort(key=lambda x: float(x.split(':')[-1].strip()))
        return data_list
    
    else:
        print(f"Error: {response.status_code} - {response.text}")

data = get_places_data()


embedding=HuggingFaceHubEmbeddings(huggingfacehub_api_token=hf_api_token,)
vectorstore = FAISS.from_texts(
    texts=data,
    embedding=embedding
)

retriever = vectorstore.as_retriever()

llm = HuggingFaceHub(
    repo_id=repo_id,
    huggingfacehub_api_token=hf_api_token,
)


template = """Answer the following question based on the context:
{context}
Question: {question}.
"""
prompt = ChatPromptTemplate.from_template(template)

# RAG
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | llm
    | StrOutputParser()
)
