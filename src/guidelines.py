import json
import os
from typing import List, Dict, Optional

class ClinicalGuidelines:
    def __init__(self, guidelines_path: str = "guidelines_db/russian_guidelines.json"):
        # Проверяем существование файла
        if not os.path.exists(guidelines_path):
            print(f"⚠️ Файл {guidelines_path} не найден. Используем пустую базу.")
            self.guidelines = {}
            return
            
        try:
            with open(guidelines_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    print(f"⚠️ Файл {guidelines_path} пуст. Используем пустую базу.")
                    self.guidelines = {}
                else:
                    self.guidelines = json.loads(content)
                    print(f"✅ Загружено {len(self.guidelines)} категорий рекомендаций")
        except json.JSONDecodeError as e:
            print(f"❌ Ошибка в JSON файле: {e}")
            print("Используем пустую базу рекомендаций")
            self.guidelines = {}
        except Exception as e:
            print(f"❌ Ошибка при загрузке: {e}")
            self.guidelines = {}
    
    def search_by_diagnosis(self, diagnosis: str) -> Dict:
        """Поиск рекомендаций по диагнозу"""
        diagnosis_lower = diagnosis.lower()
        
        # Прямое совпадение
        if diagnosis_lower in self.guidelines:
            return self.guidelines[diagnosis_lower]
        
        # Поиск по частичному совпадению
        for key in self.guidelines:
            if key in diagnosis_lower:
                return self.guidelines[key]
        
        # Поиск по ключевым словам
        keywords = {
            'легк': 'рак_легкого',
            'молоч': 'рак_молочной_железы',
            'меланом': 'меланома',
            'поджелуд': 'рак_поджелудочной_железы'
        }
        
        for word, category in keywords.items():
            if word in diagnosis_lower:
                if category in self.guidelines:
                    return self.guidelines[category]
        
        return {}
    
    def get_treatment_recommendation(self, 
                                     diagnosis: str, 
                                     line: str = "", 
                                     molecular_markers: Optional[Dict] = None) -> List[Dict]:
        """Получить рекомендации по лечению с источниками"""
        recommendations = []
        guidelines = self.search_by_diagnosis(diagnosis)
        
        if not guidelines or 'treatment_lines' not in guidelines:
            return recommendations
        
        treatment_lines = guidelines['treatment_lines']
        
        # Учитываем молекулярные маркеры
        if molecular_markers:
            # Проверяем BRAF мутацию
            if 'BRAF' in str(molecular_markers) and 'V600E' in str(molecular_markers):
                if 'BRAF_V600E' in treatment_lines:
                    braf_data = treatment_lines['BRAF_V600E']
                    if 'после_иммунотерапии' in braf_data:
                        rec = {
                            'regimen': braf_data['после_иммунотерапии'].get('regimens', []),
                            'source': braf_data['после_иммунотерапии'].get('source', ''),
                            'nccn': braf_data['после_иммунотерапии'].get('nccn', ''),
                            'evidence': 'Уровень доказательности: IA'
                        }
                        recommendations.append(rec)
            
            # Проверяем PIK3CA мутацию
            if 'PIK3CA' in str(molecular_markers):
                if 'люминальный_B' in treatment_lines:
                    luminal_data = treatment_lines['люминальный_B']
                    if 'при_мутации_PIK3CA' in luminal_data:
                        rec = {
                            'regimen': luminal_data['при_мутации_PIK3CA'].get('regimens', []),
                            'source': luminal_data['при_мутации_PIK3CA'].get('source', ''),
                            'nccn': luminal_data['при_мутации_PIK3CA'].get('nccn', ''),
                            'evidence': 'Уровень доказательности: IB'
                        }
                        recommendations.append(rec)
        
        # Добавляем рекомендации по линии терапии
        if line and line in treatment_lines:
            line_data = treatment_lines[line]
            regimens = line_data.get('regimens', [])
            if isinstance(regimens, list):
                for regimen in regimens:
                    rec = {
                        'regimen': regimen,
                        'source': line_data.get('source', ''),
                        'nccn': line_data.get('nccn', ''),
                        'indications': line_data.get('indications', ''),
                        'evidence': 'Уровень доказательности: IA'
                    }
                    recommendations.append(rec)
        
        return recommendations
    
    def format_citation(self, recommendation: Dict) -> str:
        """Форматирует ссылку для вставки в текст"""
        citation = f"Основание: {recommendation.get('source', '')}"
        if recommendation.get('nccn'):
            citation += f"; согласуется с {recommendation['nccn']}"
        if recommendation.get('evidence'):
            citation += f" ({recommendation['evidence']})"
        return citation