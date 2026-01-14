import argparse
from interactive import enhanced_interactive_mode
from mcp_orchestrator import JSONCallAnalyticsMCP

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MCP система анализа телефонных звонков')
    parser.add_argument('--mode', default='interactive',
                        choices=['interactive', 'web', 'telegram', 'api', 'test'],
                        help='Режим работы')
    parser.add_argument('--json-dir', default='./json_calls',
                        help='Путь к JSON файлам')
    parser.add_argument('--model', default='mistral-nemo:12b',  #'qwen2.5:14b', deepseek-coder:6.7b, deepseek-coder:33b
                        help='Модель Ollama')
    parser.add_argument('--telegram-token',
                        help='Токен Telegram бота (для режима telegram)')

    args = parser.parse_args()

    if args.mode == 'interactive':
        enhanced_interactive_mode(args.model)

    # elif args.mode == 'web':
    #     from api_server import run_api_server
    #
    #     run_api_server()
    #
    # elif args.mode == 'telegram':
    #     if not args.telegram_token:
    #         print("❌ Укажите --telegram-token")
    #         exit(1)
    #     from telegram_bot import run_telegram_bot
    #
    #     run_telegram_bot()
    #
    # elif args.mode == 'api':
    #     # Только API без веб-интерфейса
    #     from api_server import app
    #     import uvicorn
    #
    #     uvicorn.run(app, host="0.0.0.0", port=8000)

    elif args.mode == 'test':
        # Тестовый режим
        system = JSONCallAnalyticsMCP(args.json_dir, args.model)
        system.test_system()