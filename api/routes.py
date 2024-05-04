# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

"""
"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, EncoderDecoderModel, EncoderDecoderConfig
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("Alfahluzi/bert2bert-model99-last")
model = EncoderDecoderModel.from_pretrained("Alfahluzi/bert2bert-model99-last")
model.to(device)
"""
"""

from datetime import datetime, timezone, timedelta

from functools import wraps

from flask import request
from flask_restx import Api, Resource, fields

import jwt
import sys

from .models import db, Users, JWTTokenBlocklist
from .config import BaseConfig
import requests

rest_api = Api(version="1.0", title="Users API")


"""
    Flask-Restx models for api request and response data
"""

summarize_model = rest_api.model('Summarize', {"text": fields.String(required=True, min_length=2, max_length=512),
                                                "min_length": fields.Integer(required=False, min=1, max=256),
                                                "num_beams": fields.Integer(required=False, min=1),
                                                "repetition_penalty": fields.Float(required=False, min=1.0),
                                                "length_penalty": fields.Float(required=False, min=1.0),
                                                "no_repeat_ngram_size": fields.Float(required=False, min=1.0),
                                                "temperature": fields.Float(required=False, min=0.01),
                                                "top_k": fields.Float(required=False, min=0, max=100),
                                                "top_p": fields.Float(required=False, min=0, max=1),
                                              })


"""
   Helper function for JWT token required
"""

def token_required(f):

    @wraps(f)
    def decorator(*args, **kwargs):

        token = None

        if "authorization" in request.headers:
            token = request.headers["authorization"]

        if not token:
            return {"success": False, "msg": "Valid JWT token is missing"}, 400

        try:
            data = jwt.decode(token, BaseConfig.SECRET_KEY, algorithms=["HS256"])
            current_user = Users.get_by_email(data["email"])

            if not current_user:
                return {"success": False,
                        "msg": "Sorry. Wrong auth token. This user does not exist."}, 400

            token_expired = db.session.query(JWTTokenBlocklist.id).filter_by(jwt_token=token).scalar()

            if token_expired is not None:
                return {"success": False, "msg": "Token revoked."}, 400

            if not current_user.check_jwt_auth_active():
                return {"success": False, "msg": "Token expired."}, 400

        except:
            return {"success": False, "msg": "Token is invalid"}, 400

        return f(current_user, *args, **kwargs)

    return decorator


"""
    Flask-Restx routes
"""


@rest_api.route('/api/summarize')
class Summarize(Resource):
    """
       Creates a new user by taking 'signup_model' input
    """

    @rest_api.expect(summarize_model, validate=True)
    def post(self):

        req_data = request.get_json()

        text = req_data.get("text")
        print(text, file=sys.stdout)
        input_ids = tokenizer.encode(text, return_tensors='pt')
        summary_ids = model.generate(
                    input_ids.to(device),
                    max_length = 256, 
                    early_stopping = True,
                    use_cache = True,
                    do_sample = True,
                    min_length = req_data.get("min_length") or 10,
                    num_beams = req_data.get("num_beams") or 10,
                    repetition_penalty = req_data.get("repetition_penalty") or 2.0,
                    length_penalty = req_data.get("length_penalty") or 2.0,
                    no_repeat_ngram_size = req_data.get("no_repeat_ngram_size") or 3,
                    temperature = req_data.get("temperature") or 1.0,
                    top_k = req_data.get("top_k") or 50,
                    top_p = req_data.get("top_p") or 0.95,
        )

        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return {"success": True,
                "result": summary_text
                }, 200

