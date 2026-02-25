# flask_app.py
from flask import Flask, request, render_template_string, jsonify
import sys
import os
import json
import traceback

# Добавляем путь для импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.model import MedicalTreatmentPlanner
except ImportError:
    # Если не получается импортировать, пробуем прямой импорт
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
    from model import MedicalTreatmentPlanner

app = Flask(__name__)

# Инициализируем модель
print("="*60)
print("Инициализация медицинского AI-ассистента...")
print("="*60)

try:
    planner = MedicalTreatmentPlanner()
    print("✅ Модель успешно загружена!")
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    planner = None

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Медицинский AI-ассистент</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial; max-width: 1200px; margin: 0 auto; padding: 20px; }
            textarea { width: 100%; height: 300px; margin: 10px 0; }
            button { padding: 10px 20px; background: #4CAF50; color: white; border: none; cursor: pointer; }
            #output { white-space: pre-wrap; background: #f5f5f5; padding: 20px; margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>🏥 Медицинский AI-ассистент</h1>
        <textarea id="history" placeholder="Введите историю болезни"></textarea>
        <button onclick="generate()">Сгенерировать план</button>
        <div id="output"></div>
        
        <script>
            async function generate() {
                const history = document.getElementById('history').value;
                document.getElementById('output').innerHTML = 'Генерация...';
                
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({history: history})
                });
                
                const data = await response.json();
                document.getElementById('output').innerHTML = data.plan || data.error;
            }
        </script>
    </body>
    </html>
    """

@app.route('/generate', methods=['POST'])
def generate():
    if planner is None:
        return jsonify({'error': 'Модель не загружена'})
    
    try:
        data = request.json
        history = data.get('history', '')
        
        if not history:
            return jsonify({'error': 'Пустая история болезни'})
        
        result = planner.generate_with_citations(history)
        return jsonify({'plan': result['plan']})
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("="*60)
    print("🚀 МЕДИЦИНСКИЙ AI-АССИСТЕНТ ЗАПУЩЕН!")
    print("="*60)
    print("🌐 Открой в браузере: http://localhost:5000")
    print("📝 Для остановки нажми Ctrl+C")
    print("="*60)
    
    app.run(host='127.0.0.1', port=5000, debug=False)