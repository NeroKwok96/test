import boto3
import os
from urllib.parse import urlparse
from dotenv import load_dotenv
import psycopg2

env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)
xsdb_rds_host = os.environ['xsdb_RDS_HOST']
xsdb_rds_port = os.environ['xsdb_RDS_PORT']
xsdb_rds_database = os.environ['xsdb_RDS_DATABASE']
xsdb_rds_user = os.environ['xsdb_RDS_USER']
xsdb_rds_password = os.environ['xsdb_RDS_PASSWORD']
# XS database query
conn_to_xsdb = psycopg2.connect(
    host=xsdb_rds_host,
    port=int(xsdb_rds_port),
    database=xsdb_rds_database,
    user=xsdb_rds_user,
    password=xsdb_rds_password
)

s3_client = boto3.client("s3")
allowed_sensor_ids = []
cursor_xsdb = conn_to_xsdb.cursor()
select_xsdb_query = """select distinct node_id from sensor_history sh join sensor_location sl on sh.sensor_location_id = sl.id join sensor s on s.id = sh.sensor_id join machine m on sl.machine_id = m.id join floorplan f on f.id = m.floorplan_id join site on site.id = f.site_id join organization o on o.id = site.organization_id where site.site_id = 'tswh'"""
cursor_xsdb.execute(select_xsdb_query)
selected_rows = cursor_xsdb.fetchall()
for i in selected_rows:
    allowed_sensor_ids.append(str(i[0]))

# Function to download the S3 dataset
def download_s3_dataset(allowed_sensor_ids, local_directory):
    for sensor_id in allowed_sensor_ids:
        sensor_local_directory = os.path.join(local_directory, sensor_id)
        
        # Create the sensor directory if it doesn't exist
        if not os.path.exists(sensor_local_directory):
            os.makedirs(sensor_local_directory)
        
        url = f"https://xs-web1-data.s3.ap-southeast-1.amazonaws.com/SpiderWeb/analysed_data/emsd2_tswh/{sensor_id}/csv/vel/freq/"
        parsed_url = urlparse(url)
        bucket_name = parsed_url.netloc.split(".")[0]
        s3_key = parsed_url.path.lstrip("/")
        for year in range(2023, 2024):
            for month in range(3, 11):
                month_str = f"{year}-{month:02d}"
                response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=s3_key + month_str + "/")
                for obj in response.get("Contents", []):
                    file_name = obj["Key"]
                    local_file_path = os.path.join(sensor_local_directory, os.path.basename(file_name))
                    # Check if the file already exists
                    if os.path.exists(local_file_path):
                        print(f"File already exists: {file_name}. Skipping...")
                        continue
                    # Download the file
                    s3_client.download_file(bucket_name, file_name, local_file_path)
                    print(f"Downloaded: {file_name}")

# Download dataset
dataset_path = os.path.join(os.pardir, 'data')
download_s3_dataset(allowed_sensor_ids, dataset_path)