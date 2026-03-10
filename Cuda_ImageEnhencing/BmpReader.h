#pragma once

#pragma pack(push, 1)

struct BmpFileHeader_s
{
	unsigned short fileMarker;
	unsigned int   bufferSize;
	unsigned short unuse1;
	unsigned short unuse2;
	unsigned int   imageDataOffet;
};

struct BmpInfoHeader_s
{
    unsigned int   biSize;
    int            biWidth;
    int            biHeight;
    unsigned short biPlanes;
    unsigned short biBitCount;
    unsigned int   biCompression;
    unsigned int   biSizeImage;
    int            biXPelsPerMeter;
    int            biYPelsPerMeter;
    unsigned int   biClrUsed;
    unsigned int   biClrImportant;
};

#pragma pack(pop)


namespace Bmp
{
	bool LoadRGBs(Buf_s** rBuf, Buf_s** gBuf, Buf_s** bBuf, unsigned int& width, unsigned int& height);
	bool StoreRGBs(const char* storeName, unsigned int width, unsigned int heght, Buf_s* rBuf, Buf_s* gBuf, Buf_s* bBuf);
}