import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
import os

def get_experiment_ids(path):
    experiment_ids = set()
    for root, _, files in os.walk(path):
        for file in files:
            experiment_ids.add(os.path.basename(root))
    return experiment_ids

def check_experiments_status(results_path:str='results'):
    final = ""
    additional_infos = ""
    model_dirs = [os.path.join(results_path, model) for model in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, model))]    
    for model_dir in model_dirs:
        additional_infos += f"\n{model_dir.replace('results/', '').upper()}\n"
        experiment_ids = get_experiment_ids(model_dir)
        experiment_ids = [exp for exp in experiment_ids if "version_0" not in exp or "checkpoint-" not in exp]
        experiment_ids = [s for s in experiment_ids if s.count('_') >= 2]
        for exp in experiment_ids:
            additional_infos += f"- {exp}\n"
        final += f"Model: {model_dir.replace('results/', '')} - Experiments done: {len(experiment_ids)}\n"
    return final + f'\n--------ADDITIONAL INFOS-----------\n' +additional_infos

def send_email(subject, body, recipient_email):
    sender_email = "purificatoantonio6@gmail.com"
    password = "mhzw ypqq suuh ulgm"
    
    # Configura l'email
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()
    except Exception as e:
        print(f"Error while sending the email: {e}")

def job():
    body = check_experiments_status()
    
    send_email("Report Giornaliero Esperimenti Env Impact", body, "vineis@diag.uniroma1.it")
    send_email("Report Giornaliero Esperimenti Env Impact", body, "betello@diag.uniroma1.it")
    send_email("Report Giornaliero Esperimenti Env Impact", body, "purificato@diag.uniroma1.it")

if __name__ == "__main__":
    job()