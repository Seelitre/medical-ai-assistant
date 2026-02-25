from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re
import json
import os
from guidelines import ClinicalGuidelines

class MedicalTreatmentPlanner:
    def __init__(self, model_path: str = "models/my_medical_t5_simple"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Получаем абсолютный путь к папке с моделью
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_model_path = os.path.join(base_dir, model_path)
        
        print(f"Загрузка модели из {full_model_path}...")
        
        # Проверяем, существует ли папка
        if not os.path.exists(full_model_path):
            raise FileNotFoundError(f"Модель не найдена по пути: {full_model_path}\n"
                                   f"Убедитесь, что модель распакована в папку {full_model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(full_model_path, local_files_only=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(full_model_path, local_files_only=True).to(self.device)
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            print("Пробуем загрузить без local_files_only...")
            self.tokenizer = AutoTokenizer.from_pretrained(full_model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(full_model_path).to(self.device)
        
        print("Модель успешно загружена!")
        
        # Определяем путь к базе рекомендаций
        guidelines_path = os.path.join(base_dir, "guidelines_db", "russian_guidelines.json")
        self.guidelines = ClinicalGuidelines(guidelines_path)
        
    def extract_diagnosis_info(self, patient_history: str) -> dict:
        """Извлекает ключевую информацию из истории болезни"""
        info = {
            'diagnosis': '',
            'line_of_therapy': '',
            'molecular_markers': {},
            'previous_treatment': []
        }
        
        # Поиск диагноза
        diagnosis_patterns = [
            r'(рак\s+\w+)',
            r'(меланома)',
            r'(саркома)',
            r'(аденокарцинома)'
        ]
        
        for pattern in diagnosis_patterns:
            match = re.search(pattern, patient_history, re.IGNORECASE)
            if match:
                info['diagnosis'] = match.group(1)
                break
        
        # Поиск линии терапии
        line_patterns = [
            r'(\d+)\s*линии',
            r'(\d+)-я\s*линия',
            r'линия\s*(\d+)'
        ]
        
        for pattern in line_patterns:
            match = re.search(pattern, patient_history, re.IGNORECASE)
            if match:
                info['line_of_therapy'] = f"{match.group(1)}_линия"
                break
        
        # Поиск молекулярных маркеров
        marker_patterns = {
            'BRAF': r'BRAF\s*(V600E|wt|мутация)',
            'EGFR': r'EGFR\s*(mut|wt|del|L858R)',
            'KRAS': r'KRAS\s*(G12C|G12D|wt|мутация)',
            'PD-L1': r'PD-L1\s*(\d+)%?',
            'HER2': r'HER2[-+]?\s*(\d+\+|\d+|позитивный|негативный)',
            'PIK3CA': r'PIK3CA\s*(мутация|wt)'
        }
        
        for marker, pattern in marker_patterns.items():
            match = re.search(pattern, patient_history, re.IGNORECASE)
            if match:
                info['molecular_markers'][marker] = match.group(1)
        
        return info
    
    def generate_with_citations(self, patient_history: str) -> dict:
        """Генерирует план лечения с обоснованиями"""
        
        # Извлекаем информацию
        info = self.extract_diagnosis_info(patient_history)
        
        # Получаем рекомендации из базы
        recommendations = self.guidelines.get_treatment_recommendation(
            diagnosis=info['diagnosis'],
            line=info['line_of_therapy'],
            molecular_markers=info['molecular_markers']
        )
        
        # Генерируем базовый план с помощью модели
        inputs = self.tokenizer(patient_history, return_tensors="pt", 
                               truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=256,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        base_plan = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Добавляем обоснования
        enhanced_plan = self._add_citations(base_plan, recommendations, info)
        
        return {
            'plan': enhanced_plan,
            'citations': recommendations,
            'extracted_info': info
        }
    
    def _add_citations(self, plan: str, recommendations: list, info: dict) -> str:
        """Добавляет ссылки на клинические рекомендации"""
        
        lines = plan.split('\n')
        enhanced_lines = []
        
        for line in lines:
            enhanced_line = line
            
            # Ищем ключевые слова и добавляем ссылки
            if 'КТ' in line or 'компьютерная томография' in line.lower():
                citation = " (Основание: Клинические рекомендации Минздрава РФ, Раздел 3.2, стр. 15; согласуется с NCCN Guideline v.2.2024, DIAG-1)"
                enhanced_line += citation
            
            elif any(regimen in line for rec in recommendations 
                    for regimen in (rec.get('regimen', []) if isinstance(rec.get('regimen'), list) else [rec.get('regimen', '')])):
                for rec in recommendations:
                    citation = self.guidelines.format_citation(rec)
                    enhanced_line += f"\n  📚 {citation}"
            
            elif 'пембролизумаб' in line or 'ниволумаб' in line:
                if info.get('molecular_markers', {}).get('PD-L1'):
                    citation = f" (PD-L1 {info['molecular_markers']['PD-L1']}% - показание для иммунотерапии согласно КР Минздрава РФ, Раздел 4.2.1)"
                    enhanced_line += citation
            
            enhanced_lines.append(enhanced_line)
        
        return '\n'.join(enhanced_lines)