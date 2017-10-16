import json
import tornado.web
import tornado.gen


class MainHandler(tornado.web.RequestHandler):
    @tornado.gen.coroutine
    def prepare(self):
        if self.request.headers['Content-Type'].startswith('application/json'):
            self.json_args = json.loads(self.request.body.decode('utf-8'))
        else:
            self.json_args = None
