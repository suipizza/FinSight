import asyncio
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import nest_asyncio
from datetime import datetime, timedelta, time

# 이미 실행 중인 이벤트 루프를 허용하도록 설정
nest_asyncio.apply()

# 봇 토큰 설정
TOKEN = 'YOUR_TOKEN'

# 각 주식에 대한 현재 가격과 예측 데이터를 미리 정의합니다.
current_prices = {
    "apple": 226.05,
    "coca_cola": 69.18,
    "jpmorgan": 213.97,
    "Johnson & Johnson": 159.39
}

predictions_data = {
    "apple": [214.18, 209.28, 207.48, 206.65, 206.26],
    "coca_cola": [68.63, 66.72, 65.40, 64.65, 64.24],
    "jpmorgan": [211.91, 212.11, 211.99, 211.93, 211.92],
    "Johnson & Johnson": [160.30, 158.46, 156.09, 154.50, 153.50]
}

# 업데이트 날짜 설정 (여기에 수동으로 날짜를 입력하세요)
update_date = datetime(2024, 8, 18)  # 일요일 날짜

# 금요일 날짜 계산
friday_date = update_date - timedelta(days=2)



async def send_temp_predictions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # 메세지 시작 부분
    response = "다음주 주가 예측입니다.\n\n"

    # 각 기업에 대해 예측 데이터 추가
    for stock, current_price in current_prices.items():
        predictions = predictions_data[stock]
        stock_name = stock.capitalize()  # 주식 이름을 대문자로 시작
        response += f"{stock_name}:\n"
        response += f"금요일 가격: ${current_price:.2f}\n"

        # 각 요일과 날짜에 따른 예측 데이터 추가
        days = ["월요일", "화요일", "수요일", "목요일", "금요일"]
        for day, date, pred in zip(days, next_week_dates, predictions):
            response += f"{date.strftime('%m월 %d일')} {day}: ${pred}\n"
        response += "\n"

    # 사용자가 요청한 채널에 메시지 전송
    await update.message.reply_text(response)


# 업데이트 날짜를 기준으로 다음 주 월요일부터 금요일까지의 날짜 계산
def get_next_week_dates(update_date):
    next_monday = update_date + timedelta(days=1)  # 일요일 기준으로 다음 월요일 계산
    return [next_monday + timedelta(days=i) for i in range(5)]

next_week_dates = get_next_week_dates(update_date)

# /start 명령어에 대한 함수
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Start command received")
    # 설명 텍스트와 카테고리 선택 메시지
    message = (
        "안녕하세요! 이 봇은 주식의 향후 5일간의 예측 가격을 제공합니다.\n"
        "각 카테고리에서 관심 있는 주식을 선택하면 해당 주식의 가격 예측을 받을 수 있습니다.\n\n"
        "가격을 알고 싶은 카테고리를 선택해주세요:"
    )

    # 카테고리 버튼들 생성
    keyboard = [
        [
            InlineKeyboardButton("IT", callback_data='category_it'),
            InlineKeyboardButton("필수소비재", callback_data='category_consumer')
        ],
        [
            InlineKeyboardButton("금융", callback_data='category_finance'),
            InlineKeyboardButton("헬스케어", callback_data='category_healthcare')
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(message, reply_markup=reply_markup)

# 카테고리 선택 후 주식 목록 버튼 생성
async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    keyboard = []  # 기본적으로 빈 리스트로 초기화

    if query.data == 'category_it':
        keyboard = [
            [InlineKeyboardButton("Apple", callback_data='stock_apple')],
            [InlineKeyboardButton("뒤로가기", callback_data='back_to_categories')]
        ]
    elif query.data == 'category_consumer':
        keyboard = [
            [InlineKeyboardButton("Coca-Cola", callback_data='stock_coca_cola')],
            [InlineKeyboardButton("뒤로가기", callback_data='back_to_categories')]
        ]
    elif query.data == 'category_finance':
        keyboard = [
            [InlineKeyboardButton("JPMorgan", callback_data='stock_jpmorgan')],
            [InlineKeyboardButton("뒤로가기", callback_data='back_to_categories')]
        ]
    elif query.data == 'category_healthcare':
        keyboard = [
            [InlineKeyboardButton("Johnson & Johnson", callback_data='stock_Johnson & Johnson')],
            [InlineKeyboardButton("뒤로가기", callback_data='back_to_categories')]
        ]

    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text("주식을 선택해주세요:", reply_markup=reply_markup)

# 주식 선택 후 예측 결과 전송
async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    stock = query.data.split('_', 1)[1]  # stock_apple -> apple
    current_price = current_prices[stock]  # 금요일 가격 가져오기
    predictions = predictions_data[stock]  # 예측 데이터 가져오기

    # 각 요일과 날짜를 함께 출력
    days = ["월요일", "화요일", "수요일", "목요일", "금요일"]
    response = f"{friday_date.strftime('%m월 %d일')} 금요일 가격: ${current_price:.2f}\n\n"
    response += "다음 5일간의 예측 가격:\n"
    for day, date, pred in zip(days, next_week_dates, predictions):
        response += f"{date.strftime('%m월 %d일')} {day}: ${pred}\n"

    # "뒤로가기" 버튼 추가 (해당 카테고리의 주식 선택 화면으로 돌아가기)
    category = get_category_from_stock(stock)
    keyboard = [
        [InlineKeyboardButton("뒤로가기", callback_data=f'category_{category}')]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text(response, reply_markup=reply_markup)

# 주식에 해당하는 카테고리 반환 함수
def get_category_from_stock(stock):
    if stock in ["apple"]:
        return "it"
    elif stock in ["coca_cola"]:
        return "consumer"
    elif stock in ["jpmorgan"]:
        return "finance"
    elif stock in ["Johnson & Johnson"]:
        return "healthcare"

# 뒤로가기 버튼 핸들러
async def back_to_categories(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    # 다시 카테고리 선택 화면으로 돌아감
    keyboard = [
        [
            InlineKeyboardButton("IT", callback_data='category_it'),
            InlineKeyboardButton("필수소비재", callback_data='category_consumer')
        ],
        [
            InlineKeyboardButton("금융", callback_data='category_finance'),
            InlineKeyboardButton("헬스케어", callback_data='category_healthcare')
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text("카테고리를 선택해주세요:", reply_markup=reply_markup)


# 자동 메시지 발송 함수
async def send_weekly_predictions(context: ContextTypes.DEFAULT_TYPE):
    # 각 사용자에게 예측 데이터를 전송하는 로직
    for user_id in context.job.data['user_ids']:
        for stock, current_price in current_prices.items():
            predictions = predictions_data[stock]
            days = ["월요일", "화요일", "수요일", "목요일", "금요일"]
            response = f"{friday_date.strftime('%m월 %d일')} 금요일 가격: ${current_price:.2f}\n\n"
            response += "다음 5일간의 예측 가격:\n"
            for day, date, pred in zip(days, next_week_dates, predictions):
                response += f"{date.strftime('%m월 %d일')} {day}: ${pred}\n"
            await context.bot.send_message(chat_id=user_id, text=response)


# 봇 시작 및 명령어 핸들러 설정
async def main():
    # Application 객체 생성 (job_queue 활성화)
    application = Application.builder().token(TOKEN).build()

    # /start 및 버튼 핸들러 설정
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button, pattern='^category_'))
    application.add_handler(CallbackQueryHandler(predict, pattern='^stock_'))
    application.add_handler(CallbackQueryHandler(back_to_categories, pattern='^back_to_categories$'))
    # /temp 명령어 핸들러 추가
    application.add_handler(CommandHandler("next_week", send_temp_predictions))
    # 자동 메시지 전송을 위한 작업 추가
    user_ids = ["foxrainswap"]  # 메시지를 받을 사용자 ID 리스트를 여기에 추가 (예: [123456789, 987654321])
    application.job_queue.run_daily(send_weekly_predictions, time=time(21, 0), days=(6,), data={'user_ids': user_ids})

    # Polling을 시작하여 봇 실행
    await application.run_polling()

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
