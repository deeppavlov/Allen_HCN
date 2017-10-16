import logging
import tornado.gen
import tornado.httpclient
import tornado.web
from botserver.src.handlers.message_handler import MessageHandler

BASE_URI = '/deeppavlov/pilot/api/v1.0'
MESSAGE_URI = "{}/message".format(BASE_URI)

logger = logging.getLogger('server_api.Server_API')


class Server_API(object):
    def __init__(self):
        pass

    def make_app(self):
        return tornado.web.Application([
            (MESSAGE_URI, MessageHandler, {"app": self})
        ], debug=True)

    def start(self, port):
        logger.info('Start botserver')
        app = self.make_app()
        app.listen(port)
        tornado.ioloop.IOLoop.current().start()
