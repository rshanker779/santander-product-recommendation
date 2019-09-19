FROM python:3

ADD ./base/setup.py /developer/base/
RUN python3 /developer/base/setup.py install
ADD ./flask/setup.py /developer/flask/
RUN python3 /developer/flask/setup.py install

ADD ./base /developer/base/
ADD ./flask /developer/flask/
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN cd /developer/base && pip install . ; exit 0
ENTRYPOINT [ "python" ]

CMD [ "/developer/flask/app/app.py" ]
