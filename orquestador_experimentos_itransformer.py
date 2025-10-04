#!/usr/bin/env python3
"""
Orquestador de Experimentos iTransformer
Ejecuta experimentos, recopila métricas y genera análisis completos
Captura: Train Loss, Validation Loss, Test Loss por época
"""

import subprocess
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import re
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import argparse

class iTransformerOrchestrator:
    """Orquestador para experimentos iTransformer"""
    
    def __init__(self, base_path: str = "./", results_dir: str = "./results_analysis"):
        self.base_path = Path(base_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuración de experimentos
        self.datasets = ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2']
        self.pred_horizons = [24, 48, 96, 192, 336, 720]
        self.seq_len = 96
        
        # Parámetros de entrenamiento (basados en exp_long_term_forecasting.py)
        self.training_params = {
            'learning_rate': 0.001,
            'train_epochs': 10,
            'patience': 3,
            'batch_size': 32,
            'use_amp': False,
            'label_len': 48,
            'output_attention': False
        }
        
        # Almacenamiento de resultados
        self.results = []
        self.training_history = []
        self.experiment_logs = {}
        
    def create_experiment_script(self, dataset: str, output_path: str = None):
        """Genera script de experimento para un dataset específico"""
        
        if output_path is None:
            output_path = self.base_path / f"iTransformer_{dataset}.sh"
        
        script_content = f"""export CUDA_VISIBLE_DEVICES=1

model_name=iTransformer

"""
        
        for pred_len in self.pred_horizons:
            script_content += f"""
python -u run.py \\
  --is_training 1 \\
  --root_path ./iTransformer_datasets/ETT-small/ \\
  --data_path {dataset}.csv \\
  --model_id {dataset}_{self.seq_len}_{pred_len} \\
  --model $model_name \\
  --data {dataset} \\
  --features M \\
  --seq_len {self.seq_len} \\
  --label_len {self.training_params['label_len']} \\
  --pred_len {pred_len} \\
  --e_layers 2 \\
  --enc_in 7 \\
  --dec_in 7 \\
  --c_out 7 \\
  --des 'Exp' \\
  --d_model 128 \\
  --d_ff 128 \\
  --learning_rate {self.training_params['learning_rate']} \\
  --train_epochs {self.training_params['train_epochs']} \\
  --patience {self.training_params['patience']} \\
  --batch_size {self.training_params['batch_size']} \\
  --itr 1

"""
        
        with open(output_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(output_path, 0o755)
        print(f"✓ Script generado: {output_path}")
        return output_path
    
    def run_experiment(self, script_path: str, dataset: str) -> Dict:
        """Ejecuta un script de experimento y captura resultados"""
        
        print(f"\n{'='*70}")
        print(f"Ejecutando experimentos para {dataset}")
        print(f"{'='*70}\n")
        
        log_file = self.results_dir / f"{dataset}_execution.log"
        
        try:
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    ['bash', str(script_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )
                
                for line in process.stdout:
                    print(line, end='')
                    log.write(line)
                
                process.wait()
                
            if process.returncode == 0:
                print(f"✓ Experimento completado: {dataset}")
            else:
                print(f"✗ Error en experimento: {dataset}")
                
            return {'status': 'completed' if process.returncode == 0 else 'failed',
                    'log_file': str(log_file)}
                    
        except Exception as e:
            print(f"✗ Error ejecutando {dataset}: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def parse_training_history(self, log_file: str) -> List[Dict]:
        """Extrae historia de entrenamiento por época de los logs"""
        
        history = []
        
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Patrón para extraer información por época
        # Formato: Epoch: X, Steps: Y | Train Loss: Z.Z Vali Loss: V.V Test Loss: T.T
        epoch_pattern = r'Epoch:\s*(\d+),\s*Steps:\s*(\d+)\s*\|\s*Train Loss:\s*([0-9.]+)\s*Vali Loss:\s*([0-9.]+)\s*Test Loss:\s*([0-9.]+)'
        
        # Buscar model_id actual
        model_pattern = r'model_id[:\s]+(\w+_\d+_\d+)'
        
        # Dividir por experimentos
        experiments = re.split(r'(?=model_id)', content)
        
        for exp in experiments:
            model_match = re.search(model_pattern, exp)
            if not model_match:
                continue
                
            model_id = model_match.group(1)
            dataset, seq_len, pred_len = model_id.split('_')
            
            # Extraer todas las épocas
            epochs = re.findall(epoch_pattern, exp)
            
            for epoch_num, steps, train_loss, vali_loss, test_loss in epochs:
                history.append({
                    'dataset': dataset,
                    'model_id': model_id,
                    'seq_len': int(seq_len),
                    'pred_len': int(pred_len),
                    'epoch': int(epoch_num),
                    'steps': int(steps),
                    'train_loss': float(train_loss),
                    'vali_loss': float(vali_loss),
                    'test_loss': float(test_loss)
                })
        
        return history
    
    def parse_final_results(self, log_file: str) -> List[Dict]:
        """Extrae resultados finales (MSE, MAE) de los logs"""
        
        results = []
        
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Patrón para resultados finales del test
        # Formato: mse:X.XXXX, mae:X.XXXX
        result_pattern = r'mse:\s*([0-9.]+),\s*mae:\s*([0-9.]+)'
        model_pattern = r'model_id[:\s]+(\w+_\d+_\d+)'
        
        # Dividir por experimentos
        experiments = re.split(r'(?=model_id)', content)
        
        for exp in experiments:
            model_match = re.search(model_pattern, exp)
            if not model_match:
                continue
                
            model_id = model_match.group(1)
            dataset, seq_len, pred_len = model_id.split('_')
            
            # Buscar resultado final (último mse/mae después del test)
            # Buscar después de "test shape:"
            test_section = re.search(r'test shape:.*?(mse:\s*([0-9.]+),\s*mae:\s*([0-9.]+))', 
                                    exp, re.DOTALL)
            
            if test_section:
                mse = float(test_section.group(2))
                mae = float(test_section.group(3))
                
                results.append({
                    'dataset': dataset,
                    'seq_len': int(seq_len),
                    'pred_len': int(pred_len),
                    'model_id': model_id,
                    'mse': mse,
                    'mae': mae,
                    'rmse': np.sqrt(mse)
                })
        
        return results
    
    def parse_results_from_checkpoint_dir(self) -> List[Dict]:
        """Extrae resultados desde directorios de checkpoints y resultados"""
        
        results = []
        
        # Buscar en directorio results/
        results_dir = self.base_path / 'results'
        if results_dir.exists():
            for metrics_file in results_dir.rglob('metrics.npy'):
                try:
                    # Cargar métricas: [mae, mse, rmse, mape, mspe]
                    metrics = np.load(metrics_file)
                    
                    # Extraer información del path
                    # Formato: results/dataset_seqlen_predlen_*/metrics.npy
                    path_parts = metrics_file.parent.name.split('_')
                    
                    if len(path_parts) >= 3:
                        results.append({
                            'dataset': path_parts[0],
                            'seq_len': int(path_parts[1]) if path_parts[1].isdigit() else 96,
                            'pred_len': int(path_parts[2]) if path_parts[2].isdigit() else 0,
                            'model_id': '_'.join(path_parts[:3]),
                            'mae': float(metrics[0]),
                            'mse': float(metrics[1]),
                            'rmse': float(metrics[2]),
                            'mape': float(metrics[3]) if len(metrics) > 3 else None,
                            'mspe': float(metrics[4]) if len(metrics) > 4 else None
                        })
                except Exception as e:
                    print(f"⚠ Error leyendo {metrics_file}: {e}")
        
        # También buscar en result_long_term_forecast.txt
        result_file = self.base_path / 'result_long_term_forecast.txt'
        if result_file.exists():
            with open(result_file, 'r') as f:
                content = f.read()
            
            # Parsear resultados
            pattern = r'(\w+_\d+_\d+).*?mse:\s*([0-9.]+),\s*mae:\s*([0-9.]+)'
            matches = re.findall(pattern, content, re.DOTALL)
            
            for model_id, mse, mae in matches:
                parts = model_id.split('_')
                if len(parts) >= 3:
                    results.append({
                        'dataset': parts[0],
                        'seq_len': int(parts[1]),
                        'pred_len': int(parts[2]),
                        'model_id': model_id,
                        'mse': float(mse),
                        'mae': float(mae),
                        'rmse': np.sqrt(float(mse))
                    })
        
        return results
    
    def run_all_experiments(self):
        """Ejecuta todos los experimentos para todos los datasets"""
        
        print(f"\n{'='*70}")
        print("INICIANDO ORQUESTACIÓN DE EXPERIMENTOS")
        print(f"{'='*70}\n")
        print(f"Datasets: {', '.join(self.datasets)}")
        print(f"Horizontes de predicción: {self.pred_horizons}")
        print(f"Longitud de secuencia: {self.seq_len}")
        print(f"Épocas de entrenamiento: {self.training_params['train_epochs']}")
        print(f"Paciencia (Early Stopping): {self.training_params['patience']}\n")
        
        for dataset in self.datasets:
            # Generar script si no existe
            script_path = self.base_path / f"iTransformer_{dataset}.sh"
            if not script_path.exists():
                self.create_experiment_script(dataset, script_path)
            
            # Ejecutar experimento
            result = self.run_experiment(script_path, dataset)
            self.experiment_logs[dataset] = result
            
            # Parsear resultados
            if result['status'] == 'completed':
                # Historia de entrenamiento
                history = self.parse_training_history(result['log_file'])
                self.training_history.extend(history)
                
                # Resultados finales
                final_results = self.parse_final_results(result['log_file'])
                self.results.extend(final_results)
        
        # Intentar también parsear desde archivos de resultados
        file_results = self.parse_results_from_checkpoint_dir()
        if file_results:
            print(f"\n✓ Encontrados {len(file_results)} resultados adicionales en archivos")
            # Agregar solo si no están duplicados
            existing_ids = {r['model_id'] for r in self.results}
            for r in file_results:
                if r['model_id'] not in existing_ids:
                    self.results.append(r)
        
        # Guardar resultados
        self.save_results()
    
    def save_results(self):
        """Guarda resultados en formato CSV y JSON"""
        
        # Guardar resultados finales
        if self.results:
            df = pd.DataFrame(self.results)
            df = df.drop_duplicates(subset=['model_id'])
            
            csv_path = self.results_dir / 'results_summary.csv'
            df.to_csv(csv_path, index=False)
            print(f"\n✓ Resultados finales guardados: {csv_path}")
            
            json_path = self.results_dir / 'results_summary.json'
            with open(json_path, 'w') as f:
                json.dump(self.results, f, indent=2)
        else:
            print("⚠ No hay resultados finales para guardar")
            df = None
        
        # Guardar historia de entrenamiento
        if self.training_history:
            df_history = pd.DataFrame(self.training_history)
            history_path = self.results_dir / 'training_history.csv'
            df_history.to_csv(history_path, index=False)
            print(f"✓ Historia de entrenamiento guardada: {history_path}")
            
            history_json = self.results_dir / 'training_history.json'
            with open(history_json, 'w') as f:
                json.dump(self.training_history, f, indent=2)
        else:
            print("⚠ No hay historia de entrenamiento para guardar")
            df_history = None
        
        return df, df_history
    
    def generate_analysis(self, df: pd.DataFrame = None, df_history: pd.DataFrame = None):
        """Genera análisis completo con visualizaciones"""
        
        # Cargar datos si no se proporcionan
        if df is None:
            csv_path = self.results_dir / 'results_summary.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path)
            else:
                print("⚠ No hay datos finales para analizar")
        
        if df_history is None:
            history_path = self.results_dir / 'training_history.csv'
            if history_path.exists():
                df_history = pd.read_csv(history_path)
            else:
                print("⚠ No hay historia de entrenamiento para analizar")
        
        print(f"\n{'='*70}")
        print("GENERANDO ANÁLISIS DE RESULTADOS")
        print(f"{'='*70}\n")
        
        # Configurar estilo
        sns.set_style("whitegrid")
        
        # Análisis de resultados finales
        if df is not None and len(df) > 0:
            self._plot_error_by_horizon(df)
            self._plot_dataset_comparison(df)
            self._plot_performance_heatmap(df)
            self._generate_best_results_table(df)
        
        # Análisis de historia de entrenamiento
        if df_history is not None and len(df_history) > 0:
            self._plot_training_curves(df_history)
            self._plot_convergence_analysis(df_history)
            self._analyze_overfitting(df_history)
        
        # NUEVO: Análisis de predicciones
        self._analyze_predictions(df)
        
        # Reporte estadístico combinado
        if df is not None and len(df) > 0:
            self._generate_statistical_report(df, df_history)
        
        print(f"\n✓ Análisis completo generado en: {self.results_dir}")
    
    def _plot_training_curves(self, df_history: pd.DataFrame):
        """Gráfico de curvas de entrenamiento"""
        
        datasets = df_history['dataset'].unique()
        pred_lens = sorted(df_history['pred_len'].unique())
        
        # Seleccionar algunos horizontes representativos
        representative_horizons = [24, 96, 336, 720]
        available_horizons = [h for h in representative_horizons if h in pred_lens]
        
        if not available_horizons:
            available_horizons = pred_lens[:4]  # Primeros 4 si no hay los representativos
        
        fig, axes = plt.subplots(len(datasets), len(available_horizons), 
                                figsize=(5*len(available_horizons), 4*len(datasets)))
        
        if len(datasets) == 1:
            axes = axes.reshape(1, -1)
        if len(available_horizons) == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle('Curvas de Entrenamiento por Dataset y Horizonte', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        for i, dataset in enumerate(sorted(datasets)):
            for j, pred_len in enumerate(available_horizons):
                ax = axes[i, j]
                
                data = df_history[
                    (df_history['dataset'] == dataset) & 
                    (df_history['pred_len'] == pred_len)
                ]
                
                if len(data) > 0:
                    epochs = data['epoch'].values
                    ax.plot(epochs, data['train_loss'], 'o-', label='Train', linewidth=2, markersize=6)
                    ax.plot(epochs, data['vali_loss'], 's-', label='Validation', linewidth=2, markersize=6)
                    ax.plot(epochs, data['test_loss'], '^-', label='Test', linewidth=2, markersize=6)
                    
                    ax.set_xlabel('Época', fontsize=10, fontweight='bold')
                    ax.set_ylabel('Loss (MSE)', fontsize=10, fontweight='bold')
                    ax.set_title(f'{dataset} - Horizonte {pred_len}', fontsize=11)
                    ax.legend(loc='best', fontsize=9)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center', 
                           transform=ax.transAxes)
                    ax.set_title(f'{dataset} - Horizonte {pred_len}', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Curvas de entrenamiento generadas")
    
    def _plot_convergence_analysis(self, df_history: pd.DataFrame):
        """Análisis de convergencia del entrenamiento"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análisis de Convergencia del Entrenamiento', 
                    fontsize=16, fontweight='bold')
        
        # 1. Mejora relativa por época (promedio)
        ax = axes[0, 0]
        for dataset in sorted(df_history['dataset'].unique()):
            data = df_history[df_history['dataset'] == dataset].groupby('epoch').agg({
                'vali_loss': 'mean'
            }).reset_index()
            
            if len(data) > 1:
                # Calcular mejora relativa
                improvement = (data['vali_loss'].iloc[0] - data['vali_loss']) / data['vali_loss'].iloc[0] * 100
                ax.plot(data['epoch'], improvement, marker='o', label=dataset, linewidth=2)
        
        ax.set_xlabel('Época', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mejora Relativa (%)', fontsize=12, fontweight='bold')
        ax.set_title('Mejora Relativa en Validation Loss', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Gap Train-Validation
        ax = axes[0, 1]
        for dataset in sorted(df_history['dataset'].unique()):
            data = df_history[df_history['dataset'] == dataset].groupby('epoch').agg({
                'train_loss': 'mean',
                'vali_loss': 'mean'
            }).reset_index()
            
            if len(data) > 0:
                gap = data['vali_loss'] - data['train_loss']
                ax.plot(data['epoch'], gap, marker='s', label=dataset, linewidth=2)
        
        ax.set_xlabel('Época', fontsize=12, fontweight='bold')
        ax.set_ylabel('Gap (Vali - Train)', fontsize=12, fontweight='bold')
        ax.set_title('Diferencia Validation-Train Loss', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # 3. Velocidad de convergencia
        ax = axes[1, 0]
        for dataset in sorted(df_history['dataset'].unique()):
            data = df_history[df_history['dataset'] == dataset].groupby('epoch').agg({
                'vali_loss': 'mean'
            }).reset_index()
            
            if len(data) > 1:
                # Calcular derivada (cambio en loss)
                delta = np.diff(data['vali_loss'])
                ax.plot(data['epoch'].iloc[1:], delta, marker='o', label=dataset, linewidth=2)
        
        ax.set_xlabel('Época', fontsize=12, fontweight='bold')
        ax.set_ylabel('Δ Validation Loss', fontsize=12, fontweight='bold')
        ax.set_title('Velocidad de Convergencia', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # 4. Distribución de loss final por horizonte
        ax = axes[1, 1]
        final_epoch = df_history.groupby('model_id')['epoch'].max().to_dict()
        final_losses = []
        
        for model_id in df_history['model_id'].unique():
            final_ep = final_epoch.get(model_id, 0)
            data = df_history[
                (df_history['model_id'] == model_id) & 
                (df_history['epoch'] == final_ep)
            ]
            if len(data) > 0:
                final_losses.append({
                    'pred_len': data['pred_len'].iloc[0],
                    'vali_loss': data['vali_loss'].iloc[0]
                })
        
        if final_losses:
            df_final = pd.DataFrame(final_losses)
            sns.boxplot(data=df_final, x='pred_len', y='vali_loss', ax=ax, palette='Set2')
            ax.set_xlabel('Horizonte de Predicción', fontsize=12, fontweight='bold')
            ax.set_ylabel('Validation Loss Final', fontsize=12, fontweight='bold')
            ax.set_title('Distribución de Loss Final por Horizonte', fontsize=13)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'convergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Análisis de convergencia generado")
    
    def _analyze_overfitting(self, df_history: pd.DataFrame):
        """Análisis de overfitting"""
        
        print("\n" + "="*70)
        print("ANÁLISIS DE OVERFITTING")
        print("="*70)
        
        report = []
        
        for dataset in sorted(df_history['dataset'].unique()):
            report.append(f"\n{dataset}:")
            
            for pred_len in sorted(df_history['pred_len'].unique()):
                data = df_history[
                    (df_history['dataset'] == dataset) & 
                    (df_history['pred_len'] == pred_len)
                ]
                
                if len(data) == 0:
                    continue
                
                # Tomar última época
                final_epoch = data['epoch'].max()
                final_data = data[data['epoch'] == final_epoch].iloc[0]
                
                train_loss = final_data['train_loss']
                vali_loss = final_data['vali_loss']
                test_loss = final_data['test_loss']
                
                # Calcular gap
                gap = vali_loss - train_loss
                gap_percent = (gap / train_loss) * 100 if train_loss > 0 else 0
                
                # Clasificar overfitting
                if gap_percent < 10:
                    status = "✓ Buen ajuste"
                elif gap_percent < 25:
                    status = "⚠ Posible overfitting leve"
                else:
                    status = "✗ Overfitting significativo"
                
                report.append(f"  Horizonte {pred_len}:")
                report.append(f"    Train Loss: {train_loss:.6f}")
                report.append(f"    Vali Loss:  {vali_loss:.6f}")
                report.append(f"    Test Loss:  {test_loss:.6f}")
                report.append(f"    Gap: {gap:.6f} ({gap_percent:.2f}%)")
                report.append(f"    Estado: {status}")
        
        report_text = "\n".join(report)
        print(report_text)
        
        # Guardar reporte
        overfitting_path = self.results_dir / 'overfitting_analysis.txt'
        with open(overfitting_path, 'w') as f:
            f.write(report_text)
        
        print(f"\n✓ Análisis de overfitting guardado: {overfitting_path}")
    
    def _analyze_predictions(self, df: pd.DataFrame = None):
        """Analiza y visualiza predicciones del conjunto de prueba"""
        
        print("\n" + "="*70)
        print("ANALIZANDO PREDICCIONES DEL CONJUNTO DE PRUEBA")
        print("="*70)
        
        results_base = self.base_path / 'results'
        
        if not results_base.exists():
            print("⚠ Directorio de resultados no encontrado")
            return
        
        # Buscar archivos de predicción
        pred_files = list(results_base.rglob('pred.npy'))
        
        if not pred_files:
            print("⚠ No se encontraron archivos de predicción")
            return
        
        print(f"✓ Encontrados {len(pred_files)} archivos de predicción\n")
        
        # Crear directorio para visualizaciones de predicción
        pred_viz_dir = self.results_dir / 'prediction_analysis'
        pred_viz_dir.mkdir(exist_ok=True)
        
        # Analizar cada experimento
        predictions_summary = []
        
        for pred_file in pred_files[:12]:  # Limitar a 12 para no sobrecargar
            true_file = pred_file.parent / 'true.npy'
            
            if not true_file.exists():
                continue
            
            try:
                # Cargar datos
                pred = np.load(pred_file)
                true = np.load(true_file)
                
                # Extraer información del path
                model_id = pred_file.parent.name
                parts = model_id.split('_')
                
                if len(parts) < 3:
                    continue
                
                dataset = parts[0]
                pred_len = int(parts[2]) if parts[2].isdigit() else 0
                
                # Calcular métricas
                mae = np.mean(np.abs(pred - true))
                mse = np.mean((pred - true) ** 2)
                rmse = np.sqrt(mse)
                
                # Calcular MAPE (evitando división por cero)
                mask = np.abs(true) > 1e-8
                mape = np.mean(np.abs((pred[mask] - true[mask]) / true[mask])) * 100 if mask.any() else 0
                
                predictions_summary.append({
                    'model_id': model_id,
                    'dataset': dataset,
                    'pred_len': pred_len,
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'mape': mape,
                    'shape': pred.shape
                })
                
                # Visualizar predicciones
                self._plot_prediction_comparison(pred, true, model_id, pred_viz_dir)
                
                print(f"✓ Procesado: {model_id}")
                
            except Exception as e:
                print(f"✗ Error procesando {pred_file.name}: {e}")
        
        if predictions_summary:
            # Guardar resumen
            df_pred = pd.DataFrame(predictions_summary)
            pred_summary_path = self.results_dir / 'predictions_summary.csv'
            df_pred.to_csv(pred_summary_path, index=False)
            print(f"\n✓ Resumen de predicciones guardado: {pred_summary_path}")
            
            # Generar visualizaciones comparativas
            self._plot_prediction_errors_analysis(df_pred, pred_viz_dir)
        
        print(f"\n✓ Visualizaciones de predicción guardadas en: {pred_viz_dir}")
    
    def _plot_prediction_comparison(self, pred: np.ndarray, true: np.ndarray, 
                                   model_id: str, output_dir: Path):
        """Genera gráficos comparativos de predicción vs real"""
        
        # Seleccionar algunas muestras representativas
        n_samples = min(4, pred.shape[0])
        sample_indices = np.linspace(0, pred.shape[0]-1, n_samples, dtype=int)
        
        # Seleccionar variable representativa (última columna típicamente)
        var_idx = -1
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Predicciones vs Real - {model_id}', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for i, idx in enumerate(sample_indices):
            if i >= 4:
                break
                
            ax = axes[i]
            
            # Extraer datos
            pred_sample = pred[idx, :, var_idx]
            true_sample = true[idx, :, var_idx]
            time_steps = np.arange(len(pred_sample))
            
            # Graficar
            ax.plot(time_steps, true_sample, 'b-', label='Real', linewidth=2, alpha=0.7)
            ax.plot(time_steps, pred_sample, 'r--', label='Predicción', linewidth=2, alpha=0.7)
            ax.fill_between(time_steps, pred_sample, true_sample, alpha=0.2, color='gray')
            
            # Calcular error para esta muestra
            mae_sample = np.mean(np.abs(pred_sample - true_sample))
            
            ax.set_xlabel('Paso de Tiempo', fontsize=11, fontweight='bold')
            ax.set_ylabel('Valor', fontsize=11, fontweight='bold')
            ax.set_title(f'Muestra {idx} | MAE: {mae_sample:.4f}', fontsize=12)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / f'{model_id}_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_prediction_errors_analysis(self, df_pred: pd.DataFrame, output_dir: Path):
        """Análisis de errores de predicción"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Análisis de Errores de Predicción', fontsize=16, fontweight='bold')
        
        # 1. Distribución de MAE por horizonte
        ax = axes[0, 0]
        df_sorted = df_pred.sort_values('pred_len')
        sns.boxplot(data=df_sorted, x='pred_len', y='mae', ax=ax, palette='Set2')
        ax.set_xlabel('Horizonte de Predicción', fontsize=11, fontweight='bold')
        ax.set_ylabel('MAE', fontsize=11, fontweight='bold')
        ax.set_title('Distribución de MAE por Horizonte', fontsize=12)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. Distribución de MAPE por horizonte
        ax = axes[0, 1]
        if 'mape' in df_pred.columns:
            sns.boxplot(data=df_sorted, x='pred_len', y='mape', ax=ax, palette='Set3')
            ax.set_xlabel('Horizonte de Predicción', fontsize=11, fontweight='bold')
            ax.set_ylabel('MAPE (%)', fontsize=11, fontweight='bold')
            ax.set_title('Distribución de MAPE por Horizonte', fontsize=12)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
        
        # 3. MAE vs RMSE
        ax = axes[0, 2]
        for dataset in df_pred['dataset'].unique():
            data = df_pred[df_pred['dataset'] == dataset]
            ax.scatter(data['mae'], data['rmse'], label=dataset, s=100, alpha=0.6)
        ax.set_xlabel('MAE', fontsize=11, fontweight='bold')
        ax.set_ylabel('RMSE', fontsize=11, fontweight='bold')
        ax.set_title('Correlación MAE vs RMSE', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Línea de referencia MAE=RMSE
        max_val = max(df_pred['mae'].max(), df_pred['rmse'].max())
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='MAE=RMSE')
        
        # 4. Error promedio por dataset
        ax = axes[1, 0]
        avg_mae = df_pred.groupby('dataset')['mae'].mean().sort_values()
        colors = sns.color_palette("viridis", len(avg_mae))
        ax.barh(avg_mae.index, avg_mae.values, color=colors)
        ax.set_xlabel('MAE Promedio', fontsize=11, fontweight='bold')
        ax.set_title('Error Promedio por Dataset', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        
        # 5. Evolución del error con horizonte
        ax = axes[1, 1]
        for dataset in sorted(df_pred['dataset'].unique()):
            data = df_pred[df_pred['dataset'] == dataset].sort_values('pred_len')
            if len(data) > 1:
                ax.plot(data['pred_len'], data['mae'], marker='o', 
                       label=dataset, linewidth=2, markersize=8)
        ax.set_xlabel('Horizonte de Predicción', fontsize=11, fontweight='bold')
        ax.set_ylabel('MAE', fontsize=11, fontweight='bold')
        ax.set_title('Evolución del Error con Horizonte', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Tabla de métricas
        ax = axes[1, 2]
        ax.axis('off')
        
        # Crear tabla de resumen
        summary_data = []
        for dataset in sorted(df_pred['dataset'].unique()):
            data = df_pred[df_pred['dataset'] == dataset]
            summary_data.append([
                dataset,
                f"{data['mae'].mean():.4f}",
                f"{data['mse'].mean():.4f}",
                f"{data['rmse'].mean():.4f}"
            ])
        
        table = ax.table(cellText=summary_data,
                        colLabels=['Dataset', 'MAE', 'MSE', 'RMSE'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Estilo de la tabla
        for i in range(len(summary_data) + 1):
            for j in range(4):
                cell = table[(i, j)]
                if i == 0:
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax.set_title('Resumen de Métricas', fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        output_path = output_dir / 'prediction_errors_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Análisis de errores guardado: {output_path}")
    
    def _plot_error_by_horizon(self, df: pd.DataFrame):
        """Gráfico de evolución del error por horizonte"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Evolución del Error por Horizonte de Predicción', 
                    fontsize=16, fontweight='bold')
        
        metrics = ['mae', 'mse', 'rmse']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            for dataset in sorted(df['dataset'].unique()):
                data = df[df['dataset'] == dataset].sort_values('pred_len')
                if len(data) > 0:
                    ax.plot(data['pred_len'], data[metric], marker='o', linewidth=2, 
                        label=dataset, markersize=8)
            
            ax.set_xlabel('Horizonte de Predicción', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
            ax.set_title(f'{metric.upper()} vs Horizonte de Predicción', fontsize=13)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Análisis de tendencia normalizada
        ax = axes[1, 1]
        for dataset in sorted(df['dataset'].unique()):
            data = df[df['dataset'] == dataset].sort_values('pred_len')
            if len(data) > 1 and data['mae'].max() > data['mae'].min():
                normalized_mae = (data['mae'] - data['mae'].min()) / (data['mae'].max() - data['mae'].min())
                ax.plot(data['pred_len'], normalized_mae, marker='s', linewidth=2, 
                    label=dataset, markersize=8)
        
        ax.set_xlabel('Horizonte de Predicción', fontsize=12, fontweight='bold')
        ax.set_ylabel('MAE Normalizado', fontsize=12, fontweight='bold')
        ax.set_title('Tendencia de Error Normalizada', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'error_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Gráfico de evolución del error generado")
    
    def _plot_dataset_comparison(self, df: pd.DataFrame):
        """Gráfico de comparación entre datasets"""
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Comparación de Performance entre Datasets', 
                    fontsize=16, fontweight='bold')
        
        # MAE promedio por dataset
        avg_mae = df.groupby('dataset')['mae'].mean().sort_values()
        colors = sns.color_palette("viridis", len(avg_mae))
        axes[0].barh(avg_mae.index, avg_mae.values, color=colors)
        axes[0].set_xlabel('MAE Promedio', fontsize=12, fontweight='bold')
        axes[0].set_title('MAE Promedio por Dataset', fontsize=13)
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Box plot de MAE
        df_sorted = df.sort_values('dataset')
        sns.boxplot(data=df_sorted, y='dataset', x='mae', ax=axes[1], palette="Set2")
        axes[1].set_xlabel('MAE', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Dataset', fontsize=12, fontweight='bold')
        axes[1].set_title('Distribución de MAE por Dataset', fontsize=13)
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'dataset_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Gráfico de comparación entre datasets generado")
    
    def _plot_performance_heatmap(self, df: pd.DataFrame):
        """Heatmap de performance"""
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Heatmap de Performance', fontsize=16, fontweight='bold')
        
        # Heatmap de MAE
        pivot_mae = df.pivot_table(values='mae', index='dataset', columns='pred_len')
        sns.heatmap(pivot_mae, annot=True, fmt='.4f', cmap='RdYlGn_r', ax=axes[0], 
                cbar_kws={'label': 'MAE'})
        axes[0].set_title('MAE por Dataset y Horizonte', fontsize=13)
        axes[0].set_xlabel('Horizonte de Predicción', fontsize=12)
        axes[0].set_ylabel('Dataset', fontsize=12)
        
        # Heatmap de MSE
        pivot_mse = df.pivot_table(values='mse', index='dataset', columns='pred_len')
        sns.heatmap(pivot_mse, annot=True, fmt='.4f', cmap='RdYlGn_r', ax=axes[1],
                cbar_kws={'label': 'MSE'})
        axes[1].set_title('MSE por Dataset y Horizonte', fontsize=13)
        axes[1].set_xlabel('Horizonte de Predicción', fontsize=12)
        axes[1].set_ylabel('Dataset', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Heatmap de performance generado")
    
    def _generate_statistical_report(self, df: pd.DataFrame, df_history: pd.DataFrame = None):
        """Genera reporte estadístico detallado"""
        
        report = []
        report.append("="*70)
        report.append("REPORTE ESTADÍSTICO DE EXPERIMENTOS iTransformer")
        report.append("="*70)
        report.append(f"\nFecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total de experimentos: {len(df)}")
        report.append(f"Datasets analizados: {', '.join(sorted(df['dataset'].unique()))}")
        report.append(f"Horizontes de predicción: {sorted(df['pred_len'].unique())}")
        report.append(f"Épocas de entrenamiento: {self.training_params['train_epochs']}")
        report.append(f"Learning rate: {self.training_params['learning_rate']}")
        
        report.append("\n" + "-"*70)
        report.append("ESTADÍSTICAS GLOBALES")
        report.append("-"*70)
        
        for metric in ['mae', 'mse', 'rmse']:
            report.append(f"\n{metric.upper()}:")
            report.append(f"  Media: {df[metric].mean():.6f}")
            report.append(f"  Mediana: {df[metric].median():.6f}")
            report.append(f"  Desv. Estándar: {df[metric].std():.6f}")
            report.append(f"  Mínimo: {df[metric].min():.6f}")
            report.append(f"  Máximo: {df[metric].max():.6f}")
        
        report.append("\n" + "-"*70)
        report.append("ESTADÍSTICAS POR DATASET")
        report.append("-"*70)
        
        for dataset in sorted(df['dataset'].unique()):
            data = df[df['dataset'] == dataset]
            report.append(f"\n{dataset}:")
            report.append(f"  Experimentos: {len(data)}")
            report.append(f"  MAE: {data['mae'].mean():.6f} ± {data['mae'].std():.6f}")
            report.append(f"  MSE: {data['mse'].mean():.6f} ± {data['mse'].std():.6f}")
            report.append(f"  RMSE: {data['rmse'].mean():.6f} ± {data['rmse'].std():.6f}")
        
        report.append("\n" + "-"*70)
        report.append("ESTADÍSTICAS POR HORIZONTE")
        report.append("-"*70)
        
        for horizon in sorted(df['pred_len'].unique()):
            data = df[df['pred_len'] == horizon]
            report.append(f"\nHorizonte {horizon}:")
            report.append(f"  Experimentos: {len(data)}")
            report.append(f"  MAE: {data['mae'].mean():.6f} ± {data['mae'].std():.6f}")
            report.append(f"  MSE: {data['mse'].mean():.6f} ± {data['mse'].std():.6f}")
        
        # Correlación
        report.append("\n" + "-"*70)
        report.append("ANÁLISIS DE CORRELACIÓN")
        report.append("-"*70)
        
        corr_mae = df['pred_len'].corr(df['mae'])
        corr_mse = df['pred_len'].corr(df['mse'])
        
        report.append(f"\nCorrelación Horizonte-MAE: {corr_mae:.4f}")
        report.append(f"Correlación Horizonte-MSE: {corr_mse:.4f}")
        
        if corr_mae > 0.7:
            report.append("  → Alta correlación positiva")
        elif corr_mae > 0.3:
            report.append("  → Correlación moderada")
        else:
            report.append("  → Correlación baja")
        
        # Información de entrenamiento si está disponible
        if df_history is not None and len(df_history) > 0:
            report.append("\n" + "-"*70)
            report.append("ESTADÍSTICAS DE ENTRENAMIENTO")
            report.append("-"*70)
            
            total_epochs = df_history.groupby('model_id')['epoch'].max()
            report.append(f"\nÉpocas promedio por modelo: {total_epochs.mean():.1f}")
            report.append(f"Épocas mínimas: {total_epochs.min()}")
            report.append(f"Épocas máximas: {total_epochs.max()}")
            
            # Convergencia
            final_losses = df_history.loc[df_history.groupby('model_id')['epoch'].idxmax()]
            avg_improvement = {}
            
            for model_id in df_history['model_id'].unique():
                model_data = df_history[df_history['model_id'] == model_id].sort_values('epoch')
                if len(model_data) > 1:
                    initial = model_data.iloc[0]['vali_loss']
                    final = model_data.iloc[-1]['vali_loss']
                    improvement = (initial - final) / initial * 100
                    avg_improvement[model_id] = improvement
            
            if avg_improvement:
                report.append(f"\nMejora promedio en Validation Loss: {np.mean(list(avg_improvement.values())):.2f}%")
        
        report_text = "\n".join(report)
        
        # Guardar reporte
        report_path = self.results_dir / 'statistical_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print("\n" + report_text)
        print(f"\n✓ Reporte estadístico guardado: {report_path}")
    
    def _generate_best_results_table(self, df: pd.DataFrame):
        """Genera tabla con los mejores resultados"""
        
        print("\n" + "="*70)
        print("TOP 10 MEJORES RESULTADOS (por MAE)")
        print("="*70)
        
        best = df.nsmallest(10, 'mae')[['dataset', 'seq_len', 'pred_len', 'mae', 'mse', 'rmse']]
        print(best.to_string(index=False))
        
        # Mejores por dataset
        print("\n" + "-"*70)
        print("MEJOR RESULTADO POR DATASET")
        print("-"*70)
        
        for dataset in sorted(df['dataset'].unique()):
            best_dataset = df[df['dataset'] == dataset].nsmallest(1, 'mae')
            if not best_dataset.empty:
                row = best_dataset.iloc[0]
                print(f"\n{dataset}:")
                print(f"  Horizonte: {row['pred_len']}")
                print(f"  MAE: {row['mae']:.6f}")
                print(f"  MSE: {row['mse']:.6f}")
                print(f"  RMSE: {row['rmse']:.6f}")
        
        # Guardar tabla
        table_path = self.results_dir / 'best_results.csv'
        best.to_csv(table_path, index=False)
        print(f"\n✓ Tabla de mejores resultados guardada: {table_path}")
    
    def load_existing_results(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Carga resultados existentes si los hay"""
        
        csv_path = self.results_dir / 'results_summary.csv'
        history_path = self.results_dir / 'training_history.csv'
        
        df = None
        df_history = None
        
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f"✓ Resultados finales cargados: {len(df)} experimentos")
        else:
            print("⚠ No hay resultados finales previos")
        
        if history_path.exists():
            df_history = pd.read_csv(history_path)
            print(f"✓ Historia de entrenamiento cargada: {len(df_history)} registros")
        else:
            print("⚠ No hay historia de entrenamiento previa")
        
        return df, df_history


def main():
    parser = argparse.ArgumentParser(
        description='Orquestador de Experimentos iTransformer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python orchestrator.py --mode all                    # Ejecutar todo
  python orchestrator.py --mode run --datasets ETTm1   # Solo entrenar ETTm1
  python orchestrator.py --mode analyze                # Solo analizar resultados
  python orchestrator.py --train_epochs 20 --patience 5  # Personalizar entrenamiento
        """
    )
    
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'run', 'analyze'],
                       help='Modo: all (ejecutar+analizar), run (solo ejecutar), analyze (solo analizar)')
    parser.add_argument('--base_path', type=str, default='./',
                       help='Ruta base del proyecto')
    parser.add_argument('--results_dir', type=str, default='./results_analysis',
                       help='Directorio para guardar análisis')
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2'],
                       help='Datasets a procesar')
    parser.add_argument('--train_epochs', type=int, default=10,
                       help='Número de épocas de entrenamiento')
    parser.add_argument('--patience', type=int, default=3,
                       help='Paciencia para Early Stopping')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Tamaño de batch')
    
    args = parser.parse_args()
    
    # Crear orquestador
    orchestrator = iTransformerOrchestrator(
        base_path=args.base_path,
        results_dir=args.results_dir
    )
    
    # Configurar parámetros
    orchestrator.datasets = args.datasets
    orchestrator.training_params['train_epochs'] = args.train_epochs
    orchestrator.training_params['patience'] = args.patience
    orchestrator.training_params['learning_rate'] = args.learning_rate
    orchestrator.training_params['batch_size'] = args.batch_size
    
    # Ejecutar según modo
    if args.mode in ['all', 'run']:
        orchestrator.run_all_experiments()
        df, df_history = orchestrator.save_results()
    else:
        df, df_history = orchestrator.load_existing_results()
    
    if args.mode in ['all', 'analyze']:
        orchestrator.generate_analysis(df, df_history)
    
    print("\n" + "="*70)
    print("ORQUESTACIÓN COMPLETADA")
    print("="*70)
    print(f"\nResultados guardados en: {orchestrator.results_dir}")
    print("\nArchivos generados:")
    print("  - results_summary.csv              (Métricas finales)")
    print("  - training_history.csv             (Historia de entrenamiento)")
    print("  - predictions_summary.csv          (Resumen de predicciones)")
    print("  - statistical_report.txt           (Reporte estadístico)")
    print("  - training_curves.png              (Curvas de entrenamiento)")
    print("  - convergence_analysis.png         (Análisis de convergencia)")
    print("  - error_evolution.png              (Evolución del error)")
    print("  - overfitting_analysis.txt         (Análisis de overfitting)")
    print("  - prediction_analysis/             (Visualizaciones de predicciones)")
    print("      ├── *_comparison.png           (Comparación predicción vs real)")
    print("      └── prediction_errors_analysis.png (Análisis de errores)")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()