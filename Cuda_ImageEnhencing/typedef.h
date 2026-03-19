#pragma once


struct Buf_s
{
	static const int MAX_DATA_SIZE = 2147483647;

	unsigned int size = 0;
	char data[0];
};