import pandas as pd
import ibm_boto3
from ibm_botocore.client import Config
from tempfile import NamedTemporaryFile

# initialize a cloud object store
def get_cloud_object_store(cos_arguments):
    # Open connection to cloud object storage
    cos = ibm_boto3.client("s3",
                           ibm_api_key_id=cos_arguments["apikey"],
                           ibm_service_instance_id=cos_arguments["resource_instance_id"],
                           ibm_auth_endpoint="https://iam.ng.bluemix.net/oidc/token",
                           config=Config(signature_version="oauth"),
                           endpoint_url="https://s3.us-south.objectstorage.softlayer.net")

    return cos

def dwnld_csvdf_cos(cos, bucket, key):
    # the temporary in-memory file for holding the model file
    tmpfile = NamedTemporaryFile()

    # read the model into this file
    cos.download_file(bucket, key, tmpfile.name)

    # convert to df and return
    return pd.read_csv(tmpfile.name)