# $DEL_BEGIN

FROM python:3.10.6-buster
COPY wasteye-main wasteye-main
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
#CMD uvicorn wasteye-main.api.fast:app --host 0.0.0.0 --port $PORT
#RUN ["uvicorn", "wasteye-main.api.fast:app", "--host", "0.0.0.0", "--port" ,"90","-reload" $PORT]
CMD ["uvicorn", "wasteye-main.api.fast:app", "--host", "0.0.0.0", "--port" ,"90","-reload" $PORT]
# $DEL_END