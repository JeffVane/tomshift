import sys
import os
import threading
import time
import webbrowser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from app import app as flask_app

def run_flask():
    flask_app.run(port=5055, debug=False, use_reloader=False)

flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

# Aguarda o Flask subir e abre o navegador automaticamente
time.sleep(1.5)
webbrowser.open("http://localhost:5055")

print("\n🎵 TomShift rodando em http://localhost:5055")
print("   Pressione Ctrl+C para encerrar.\n")

# Mantém o processo vivo
try:
    flask_thread.join()
except KeyboardInterrupt:
    print("\nEncerrando TomShift...")
    sys.exit(0)