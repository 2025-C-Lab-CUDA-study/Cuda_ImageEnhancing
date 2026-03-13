#include "typedef.h"
#include "BmpReader.h"

#include <iostream>
#include <fstream>


#define BMPROWSIZE(width) ((width * 3 + 3) & (~3)) // for 4 byte alligne, remove under 2 bits

constexpr unsigned short BMP_MARKER = 0x4D42;
constexpr int FILE_MARKER_SIZE = 14;
constexpr int FILE_BODY_SIZE = 40;
constexpr int BIT_PER_PIXEL = 24;

static const char* FILE_NAME = "../lenna.bmp"; 



namespace Bmp
{
	bool LoadRGBs(Buf_s** rBuf, Buf_s** gBuf, Buf_s** bBuf, unsigned int& width, unsigned int& height)
	{
		bool result = false;


		std::ifstream file("C:/Users/james/source/2026/Cuda/Cuda_ImageEnhancing/lenna.bmp", std::ios::binary);

		if (!file)
		{
			fprintf(stderr, "file failed with error : INVALID_FILE_NAME | FILE_NOT_EXIST\n");
			goto LB_RETURN;
		}

		BmpFileHeader_s fileHeader;
		file.read(reinterpret_cast<char*>(&fileHeader), sizeof(fileHeader));
		if (fileHeader.fileMarker != BMP_MARKER)
		{
			fprintf(stderr, "read failed with error : INVALID_FILE_TYPE | NO_FILE_MARKER\n");
			goto LB_RETURN;
		}

		BmpInfoHeader_s infoHeader;
		file.read(reinterpret_cast<char*>(&infoHeader), sizeof(infoHeader));

		width = infoHeader.biWidth;
		height = infoHeader.biHeight;

		if (infoHeader.biBitCount != BIT_PER_PIXEL)
		{
			fprintf(stderr, "invalid bitPix, unsupported file format\n");
			goto LB_RETURN;
		}

		unsigned int bufferSize = width * height;

		if (bufferSize > (*rBuf)->MAX_DATA_SIZE || bufferSize == 0)
		{
			fprintf(stderr, "invalid bufferSize, unsupported buffer size\n");
			goto LB_RETURN;
		}

		*rBuf = (Buf_s*)malloc(sizeof(Buf_s) + sizeof(char) * (bufferSize - 1));
		*gBuf = (Buf_s*)malloc(sizeof(Buf_s) + sizeof(char) * (bufferSize - 1));
		*bBuf = (Buf_s*)malloc(sizeof(Buf_s) + sizeof(char) * (bufferSize - 1));
		(*rBuf)->size = (*gBuf)->size = (*bBuf)->size = bufferSize;

		unsigned int rowSize = BMPROWSIZE(width); 
		char* tempBuf = new char[rowSize];

		for (unsigned int i = 0; i < height; ++i)
		{
			file.read(reinterpret_cast<char*>(tempBuf), rowSize);

			for (unsigned int j = 0; j < width; ++j)
			{
				unsigned int idx = (height - 1 - i) * width + j;
				(*bBuf)->data[idx] = tempBuf[j * 3 + 0];
				(*gBuf)->data[idx] = tempBuf[j * 3 + 1];
				(*rBuf)->data[idx] = tempBuf[j * 3 + 2];
			}
		}

		result = true;
		
	LB_RETURN:

		if (file) {file.close();}
		return result;
	}


	bool StoreRGBs(const char* storeName, unsigned int width, unsigned int height, Buf_s* rBuf, Buf_s* gBuf, Buf_s* bBuf)
	{
		bool result = false;

		std::ofstream file(storeName, std::ios::binary);
		if (!file)
		{
			fprintf(stderr, "file failed with error : FILE_ALREADY_EXIST\n");
			goto LB_RETURN;
		}

		unsigned int rowSize = BMPROWSIZE(width);
		unsigned int imageSize = rowSize * height;
		

		BmpFileHeader_s fileHeader;
		fileHeader.fileMarker = BMP_MARKER;
		fileHeader.bufferSize = FILE_MARKER_SIZE + FILE_BODY_SIZE + imageSize;
		fileHeader.imageDataOffet = FILE_MARKER_SIZE + FILE_BODY_SIZE;

		file.write(reinterpret_cast<char*>(&fileHeader), sizeof(fileHeader));


		BmpInfoHeader_s infoHeader;
		infoHeader.biSize = FILE_BODY_SIZE;
		infoHeader.biWidth = width;
		infoHeader.biHeight = height;
		infoHeader.biPlanes = 1;
		infoHeader.biBitCount = BIT_PER_PIXEL;
		infoHeader.biCompression= 0; // no compression
		infoHeader.biSizeImage = imageSize;
		infoHeader.biXPelsPerMeter = 0;
		infoHeader.biYPelsPerMeter = 0;
		infoHeader.biClrUsed = 0;
		infoHeader.biClrImportant = 0;

		file.write(reinterpret_cast<char*>(&infoHeader), sizeof(infoHeader));


		char* tempBuf = new char[rowSize];

		for (unsigned int i = 0; i < height; ++i)
		{
			for (unsigned int j = 0; j < width; ++j)
			{
				unsigned int idx = (height - 1 - i) * width + j;
				tempBuf[j * 3 + 0] = bBuf->data[idx];
				tempBuf[j * 3 + 1] = gBuf->data[idx];
				tempBuf[j * 3 + 2] = rBuf->data[idx];
			}

			file.write(reinterpret_cast<char*>(tempBuf), rowSize);
		}

		result = true;

	LB_RETURN:

		if (file) { file.close(); }

		return result;
	}
}