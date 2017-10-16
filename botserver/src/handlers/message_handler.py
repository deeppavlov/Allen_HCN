import json
import logging
import tornado.gen
import tornado.web
from botserver.src.handlers.main_handler import MainHandler


class MessageHandler(MainHandler):
    def initialize(self, app):
        self.app = app
        self.logger = logging.getLogger('server_api.PredictMessageHandler')

    @tornado.gen.coroutine
    @tornado.web.asynchronous
    def post(self):
        response = {'response': None}
        if self.json_args:
            try:
                # TODO this is a temporary response, should be meaninful.
                r = "Any random response"
            except Exception:
                r = "Internal botserver error"
                self.logger.exception(r)
                self.write(json.dumps(r, ensure_ascii=False))
                self.finish()
                return
        else:
            r = "No json data is provided."
        response['response'] = r
        self.write(json.dumps(response, ensure_ascii=False))
        self.finish()
