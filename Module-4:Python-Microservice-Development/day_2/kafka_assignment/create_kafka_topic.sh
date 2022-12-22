#!/bin/bash

docker-compose exec broker kafka-topics --create --topic kafka_event --bootstrap-server broker:9092
