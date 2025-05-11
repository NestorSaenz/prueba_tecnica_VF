
from openai import AzureOpenAI
import os
import pandas as pd
from typing import Dict, List

class ChurnMessageGenerator:
    def __init__(self, openai_config: Dict):
        """
        Generador de mensajes basado en segmentos de churn
        
        Args:
            openai_config: Dict con 'api_key' y 'endpoint'
        """
        self.client = AzureOpenAI(
            api_key=openai_config.get('api_key'),
            api_version="2023-05-15",
            azure_endpoint=openai_config.get('endpoint')
        )
        
        # Plantillas por segmento
        self.templates = {
            "alto_valor": """
            Como miembro valioso (engagement {engagement:.1f}/10), te ofrecemos:
            - {action_1}
            - {action_2}
            Mensaje clave: {benefit} 🎯
            """,
            "medio": """
            Hemos seleccionado para ti: {action_1} porque tu engagement es {engagement:.1f}/10.
            Beneficio: {benefit} 📈
            """,
            "bajo": """
            ¡No te pierdas {action_1}! Ideal para reactivar tu aprendizaje.
            Ventaja: {benefit} 🔄
            """
        }
        
        self.actions = {
            "alto_valor": [
                "sesión estratégica personalizada",
                "acceso prioritario a nuevos cursos"
            ],
            "medio": [
                "programa de mejora continua",
                "webinars exclusivos"
            ],
            "bajo": [
                "curso de introducción",
                "consulta gratuita con expertos"
            ]
        }
    
    def generate_message(self, user_data: Dict) -> str:
        """
        Genera mensaje personalizado basado en segmento de churn
        
        Args:
            user_data: {
                'segment': str,
                'engagement': float,
                'churn_prob': float,
                'pc_components': List[float] (opcional)
            }
        
        Returns:
            str: Mensaje personalizado
        """
        try:
            segment = user_data["segment"]
            
            # Seleccionar plantilla y acciones
            template = self.templates.get(segment, self.templates["medio"])
            actions = self.actions.get(segment, self.actions["medio"])
            
            # Beneficio basado en probabilidad de churn
            benefit = self._get_benefit(user_data["churn_prob"], segment)
            
            # Generar con OpenAI
            prompt = f"""Genera un mensaje para segmento {segment} con:
            - Engagement: {user_data['engagement']}/10
            - Riesgo churn: {'Alto' if user_data['churn_prob'] > 0.2 else 'Medio' if user_data['churn_prob'] > 0.1 else 'Bajo'}
            - Acciones: {', '.join(actions)}
            - Beneficio clave: {benefit}
            Formato: 2 oraciones máximo, 1 emoji, tono profesional"""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Eres un asistente educativo especializado en retención."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=100
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generando mensaje: {str(e)}")
            return f"Acciones recomendadas: {actions[0]} y {actions[1]}"
    
    def _get_benefit(self, churn_prob: float, segment: str) -> str:
        """Determina el beneficio a destacar"""
        if segment == "alto_valor":
            return "Mantener tu estatus privilegiado"
        elif churn_prob > 0.2:
            return "Reducir tu riesgo de abandono"
        else:
            return "Maximizar tu experiencia de aprendizaje"