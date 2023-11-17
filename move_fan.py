import os
import pandas as pd
import numpy as np
import glob
from dotenv import load_dotenv
import psycopg2
from datetime import datetime

env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)
xswh_rds_host = os.environ['xswh_RDS_HOST']
xswh_rds_port = os.environ['xswh_RDS_PORT']
xswh_rds_database = os.environ['xswh_RDS_DATABASE']
xswh_rds_user = os.environ['xswh_RDS_USER']
xswh_rds_password = os.environ['xswh_RDS_PASSWORD']

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
cursor_xsdb = conn_to_xsdb.cursor()
select_xsdb_query = """
SELECT machine_name, s.node_id, period_from, period_to, location_name
FROM sensor_history sh
JOIN sensor_location sl ON sh.sensor_location_id = sl.id
JOIN sensor s ON s.id = sh.sensor_id
JOIN machine m ON sl.machine_id = m.id
JOIN floorplan f ON f.id = m.floorplan_id
JOIN site ON site.id = f.site_id
JOIN organization o ON o.id = site.organization_id
WHERE site.site_id = 'tswh'
GROUP BY machine_name, s.node_id, period_from, period_to, location_name
order by period_from desc;
"""

# Execute Query with machine_name_list as parameters
cursor_xsdb.execute(select_xsdb_query)

select_rows = cursor_xsdb.fetchall()
for index in range(len(select_rows)):
    machine_name = select_rows[index][0]
    sensor_id = select_rows[index][1]
    period_from = select_rows[index][2].strftime("%Y%m%d_%H%M%S")
    period_to = select_rows[index][3]
    if period_to == None: period_to = "util_now"
    else: period_to = select_rows[index][3].strftime("%Y%m%d_%H%M%S")
    location_name = select_rows[index][4]
    sensor_location_directory = os.path.join(os.pardir, 'analyzed_data', f"{machine_name}" , f"{period_from} - {period_to}", f"{location_name}", f"{sensor_id}")
    if not os.path.exists(sensor_location_directory):
        os.makedirs(sensor_location_directory)
    # print(sensor_location_directory)
csv_path = os.path.join(os.pardir, 'data')
print(os.listdir(csv_path))

# if csv's timestamp is higher than the period_from in select_rows but lower than period_to, then move the csv into the period file.