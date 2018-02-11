#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import IntEnum


class ObjectClass(IntEnum):
    UNKNOWN = 0, 'Unknown'
    DONTCARE = 1, 'DontCare'
    PASSENGER_CAR = 2, 'PassengerCar'
    PEDESTRIAN = 3, 'Pedestrian'
    VAN = 4, 'Van'
    TRUCK = 5, 'Truck'
    PERSON_SITTING = 6, 'Person_sitting'
    CYCLIST = 7, 'Cyclist'
    TRAM = 8, 'Tram'
    MISC = 9, 'Misc'
    LARGE_VEHICLE = 10, 'LargeVehicle'
    TRAILER = 11, 'Trailer'

    def __new__(cls, value, full_name):
        member = int.__new__(cls)
        member._value_ = value
        member._full_name = full_name
        return member

    def __int__(self):
        return self.value

    def __repr__(self):
        return self._full_name

    def __str__(self):
        return self._full_name
