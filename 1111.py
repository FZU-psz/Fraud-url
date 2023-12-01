# 大数据 彭诗忠
#开发时间  22:16 2023/11/24

from http.server import BaseHTTPRequestHandler, HTTPServer


class RequestHandler(BaseHTTPRequestHandler):
    def _set_response(self, status_code=200, content_type='text/plain'):
        self.send_response(status_code)
        self.send_header('Content-type', content_type)
        self.end_headers()

    def do_POST(self):
        if self.path == '/diting':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            print(post_data)
            response = 'hello world'

            self._set_response()
            self.wfile.write(response.encode('utf-8'))
        else:
            self._set_response(status_code=404)
            self.wfile.write('Not Found'.encode('utf-8'))


def run_server():
    server_address = ('', 8080)
    httpd = HTTPServer(server_address, RequestHandler)
    print('Server started on port 8080')
    httpd.serve_forever()


run_server()