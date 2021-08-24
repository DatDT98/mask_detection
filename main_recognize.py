import base64
from concurrent import futures
import numpy as np

import cv2
import grpc

from proto import recognize_server_pb2 as service, recognize_server_pb2_grpc as service_grpc
from services.facial_recognition_service import FacialRecognitionService
from services.faiss_service import FaissService
from services.feature_vector_service import FeatureVectorService
from services.milvus_service import MilvusService
from services.serving_service import ServingService
from services.storage_service import StorageService

from utils.application_properties import get_config_variable
from utils.customized_exception import BadRequestException
from utils.error_handler import handle_error_status
from utils.logging import logger
from utils import error_code

api_key = "991867b0-8833-4ee2-98f5-d5a7906433af"


def check_api_key(context):
    provided_api_key = ""
    for key, value in context.invocation_metadata():
        if key == "api_key":
            provided_api_key = str(value)
    if provided_api_key != api_key.strip():
        raise BadRequestException(error_code.UNAUTHORIZED, "api_key")


class FacialRecognitionServer(service_grpc.FaceRecognitionServicer):
    def __init__(self,
                 facial_recognition_service: FacialRecognitionService,
                 feature_vector_service: FeatureVectorService):
        self.facial_recognition_service = facial_recognition_service
        self.feature_vector_service = feature_vector_service

    def Recognize(self, request_iterator, context):
        source_url = "unknown"
        for key, value in context.invocation_metadata():
            if key == "source_url":
                source_url = str(value)
        logger.info("Receive face stream from source: {}".format(source_url))
        response_iterator = self.facial_recognition_service.recognize_stream(request_iterator)

        for response in response_iterator:
            response_list = []
            for response_element in response:
                response_list.append(service.LabeledFace(face_token=str(response_element["face_id"]),
                                                         confidence=response_element["confidence"],
                                                         track_id=response_element["track_id"]))
            yield service.RecognizeResponse(faces=response_list)
        logger.info("Stop receive face stream from source: {}".format(source_url))


if __name__ == "__main__":
    _serving_service = ServingService()
    _storage_service = StorageService()

    _feature_vector_service = MilvusService()
    recognition_service = FacialRecognitionService(_serving_service,
                                                   _feature_vector_service,
                                                   None,
                                                   _storage_service)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=25))
    recognition_server = FacialRecognitionServer(recognition_service, _feature_vector_service)
    service_grpc.add_FaceRecognitionServicer_to_server(recognition_server, server)
    server.add_insecure_port("[::]:" + str(get_config_variable("server_port2")))
    server.start()
    server.wait_for_termination()
