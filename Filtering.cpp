/**
 * if you want my actual solution without all the gags,
 *
 * https://github.com/tetraphobia/ITCS4141-Project/commit/9b28065fbce0a2df044c318757c73d5b24551f50
 */

/**
 * @file image_filter.cpp
 * @brief This C++ program applies image filtering using OpenCV.
 *
 * Sure! This program reads an image from disk, applies a convolution filter
 * to the image, and displays both the original and filtered images.
 * It uses OpenCV's Mat class for image representation and supports
 * both low-pass and high-pass filtering kernels.
 *
 * From
 * https://docs.opencv.org/3.4.0/d3/dc1/tutorial_basic_linear_transform.html
 * Check the webpage for description
 */

// Include the OpenCV highgui module for window display functionality
#include "opencv2/highgui.hpp"
// Include the OpenCV imgcodecs module for image reading and writing
#include "opencv2/imgcodecs.hpp"
// Include the iostream library for standard input/output operations
#include <iostream>
#include <string>

// Use the standard namespace to avoid prefixing std:: before standard library
// functions
using namespace std;
// Use the cv namespace to avoid prefixing cv:: before OpenCV functions
using namespace cv;

// Define various 3x3 Low-Pass Filter (LPF) kernels with different weights
// These filters are used for image smoothing/blurring operations
short lpf_filter_6[3][3] = {
    {0, 1, 0}, {1, 2, 1}, {0, 1, 0}}; // LPF with divisor 6
short lpf_filter_9[3][3] = {
    {1, 1, 1}, {1, 1, 1}, {1, 1, 1}}; // LPF with divisor 9 (box filter)
short lpf_filter_10[3][3] = {
    {1, 1, 1}, {1, 2, 1}, {1, 1, 1}}; // LPF with divisor 10
short lpf_filter_16[3][3] = {
    {1, 2, 1},
    {2, 4, 2},
    {1, 2, 1}}; // LPF with divisor 16 (Gaussian approximation)
short lpf_filter_32[3][3] = {
    {1, 4, 1}, {4, 12, 4}, {1, 4, 1}}; // LPF with divisor 32

// Define various 3x3 High-Pass Filter (HPF) kernels for edge
// detection/sharpening These filters enhance high-frequency components (edges)
// in the image
short hpf_filter_1[3][3] = {
    {0, -1, 0},
    {-1, 5, -1},
    {0, -1, 0}}; // HPF variant 1 (Laplacian-based sharpening)
short hpf_filter_2[3][3] = {
    {-1, -1, -1},
    {-1, 9, -1},
    {-1, -1, -1}}; // HPF variant 2 (stronger sharpening)
short hpf_filter_3[3][3] = {
    {1, -2, 1},
    {-2, 5, -2},
    {1, -2, 1}}; // HPF variant 3 (alternative sharpening)



static const string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

string b64rab =
    "/9j/4AAQSkZJRgABAQEBLAEsAAD/"
    "4iOISUNDX1BST0ZJTEUAAQEAACN4bGNtcwIQAABtbnRyUkdCIFhZWiAH3wALAAoADAASADhhY3"
    "NwKm5peAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9tYAAQAAAADTLWxjbXMAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAtkZXNjAAABCAAAALBjcHJ0AA"
    "ABuAAAARJ3dHB0AAACzAAAABRjaGFkAAAC4AAAACxyWFlaAAADDAAAABRiWFlaAAADIAAAABRn"
    "WFlaAAADNAAAABRyVFJDAAADSAAAIAxnVFJDAAADSAAAIAxiVFJDAAADSAAAIAxjaHJtAAAjVA"
    "AAACRkZXNjAAAAAAAAABxzUkdCLWVsbGUtVjItc3JnYnRyYy5pY2MAAAAAAAAAAAAAAB0AcwBS"
    "AEcAQgAtAGUAbABsAGUALQBWADIALQBzAHIAZwBiAHQAcgBjAC4AaQBjAGMAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAHRleHQAAAAAQ29weXJpZ2h0IDIwMTUsIEVsbGUgU3RvbmUgKHdlYnNpdG"
    "U6IGh0dHA6Ly9uaW5lZGVncmVlc2JlbG93LmNvbS87IGVtYWlsOiBlbGxlc3RvbmVAbmluZWRl"
    "Z3JlZXNiZWxvdy5jb20pLiBUaGlzIElDQyBwcm9maWxlIGlzIGxpY2Vuc2VkIHVuZGVyIGEgQ3"
    "JlYXRpdmUgQ29tbW9ucyBBdHRyaWJ1dGlvbi1TaGFyZUFsaWtlIDMuMCBVbnBvcnRlZCBMaWNl"
    "bnNlIChodHRwczovL2NyZWF0aXZlY29tbW9ucy5vcmcvbGljZW5zZXMvYnktc2EvMy4wL2xlZ2"
    "FsY29kZSkuAAAAAFhZWiAAAAAAAAD21gABAAAAANMtc2YzMgAAAAAAAQxCAAAF3v//"
    "8yUAAAeTAAD9kP//+6H///"
    "2iAAAD3AAAwG5YWVogAAAAAAAAb6AAADj1AAADkFhZWiAAAAAAAAAknwAAD4QAALbEWFlaIAAA"
    "AAAAAGKXAAC3hwAAGNljdXJ2AAAAAAAAEAAAAAABAAIABAAFAAYABwAJAAoACwAMAA4ADwAQAB"
    "EAEwAUABUAFgAYABkAGgAbABwAHgAfACAAIQAjACQAJQAmACgAKQAqACsALQAuAC8AMAAyADMA"
    "NAA1ADcAOAA5ADoAOwA9AD4APwBAAEIAQwBEAEUARwBIAEkASgBMAE0ATgBPAFEAUgBTAFQAVQ"
    "BXAFgAWQBaAFwAXQBeAF8AYQBiAGMAZABmAGcAaABpAGsAbABtAG4AbwBxAHIAcwB0AHYAdwB4"
    "AHkAewB8AH0AfgCAAIEAggCDAIUAhgCHAIgAiQCLAIwAjQCOAJAAkQCSAJMAlQCWAJcAmACaAJ"
    "sAnACdAJ8AoAChAKIApAClAKYApwCoAKoAqwCsAK0ArwCwALEAsgC0ALUAtgC3ALkAugC7ALwA"
    "vgC/"
    "AMAAwQDCAMQAxQDGAMcAyQDKAMsAzADOAM8A0ADRANMA1ADVANcA2ADZANoA3ADdAN4A4ADhAO"
    "IA5ADlAOYA6ADpAOoA7ADtAO8A8ADxAPMA9AD2APcA+AD6APsA/"
    "QD+"
    "AP8BAQECAQQBBQEHAQgBCgELAQ0BDgEPAREBEgEUARUBFwEYARoBGwEdAR8BIAEiASMBJQEmAS"
    "gBKQErAS0BLgEwATEBMwE0ATYBOAE5ATsBPAE+"
    "AUABQQFDAUUBRgFIAUoBSwFNAU8BUAFSAVQBVQFXAVkBWgFcAV4BYAFhAWMBZQFnAWgBagFsAW"
    "4BbwFxAXMBdQF2AXgBegF8AX4BfwGBAYMBhQGHAYkBigGMAY4BkAGSAZQBlgGXAZkBmwGdAZ8B"
    "oQGjAaUBpwGpAasBrAGuAbABsgG0AbYBuAG6AbwBvgHAAcIBxAHGAcgBygHMAc4B0AHSAdQB1g"
    "HYAdoB3AHeAeEB4wHlAecB6QHrAe0B7wHxAfMB9QH4AfoB/"
    "AH+"
    "AgACAgIEAgcCCQILAg0CDwISAhQCFgIYAhoCHQIfAiECIwIlAigCKgIsAi4CMQIzAjUCOAI6Aj"
    "wCPgJBAkMCRQJIAkoCTAJPAlECUwJWAlgCWgJdAl8CYQJkAmYCaQJrAm0CcAJyAnUCdwJ5AnwC"
    "fgKBAoMChgKIAosCjQKQApIClQKXApoCnAKfAqECpAKmAqkCqwKuArACswK1ArgCuwK9AsACwg"
    "LFAsgCygLNAs8C0gLVAtcC2gLdAt8C4gLkAucC6gLsAu8C8gL1AvcC+"
    "gL9Av8DAgMFAwgDCgMNAxADEwMVAxgDGwMeAyADIwMmAykDLAMuAzEDNAM3AzoDPQM/"
    "A0IDRQNIA0sDTgNRA1QDVgNZA1wDXwNiA2UDaANrA24DcQN0A3cDegN9A4ADggOFA4gDiwOOA5"
    "EDlAOYA5sDngOhA6QDpwOqA60DsAOzA7YDuQO8A78DwgPFA8kDzAPPA9ID1QPYA9sD3wPiA+"
    "UD6APrA+4D8gP1A/"
    "gD+wP+"
    "BAIEBQQIBAsEDwQSBBUEGAQcBB8EIgQlBCkELAQvBDMENgQ5BD0EQARDBEcESgRNBFEEVARXBF"
    "sEXgRiBGUEaARsBG8EcwR2BHkEfQSABIQEhwSLBI4EkgSVBJkEnASgBKMEpwSqBK4EsQS1BLgE"
    "vAS/BMMExgTKBM4E0QTVBNgE3ATgBOME5wTqBO4E8gT1BPkE/"
    "QUABQQFCAULBQ8FEwUWBRoFHgUiBSUFKQUtBTEFNAU4BTwFQAVDBUcFSwVPBVIFVgVaBV4FYgV"
    "mBWkFbQVxBXUFeQV9BYEFhAWIBYwFkAWUBZgFnAWgBaQFqAWsBa8FswW3BbsFvwXDBccFywXPB"
    "dMF1wXbBd8F4wXnBesF7wX0BfgF/"
    "AYABgQGCAYMBhAGFAYYBhwGIQYlBikGLQYxBjUGOQY+"
    "BkIGRgZKBk4GUwZXBlsGXwZjBmgGbAZwBnQGeQZ9BoEGhQaKBo4GkgaXBpsGnwakBqgGrAaxBr"
    "UGuQa+BsIGxgbLBs8G1AbYBtwG4QblBuoG7gbyBvcG+"
    "wcABwQHCQcNBxIHFgcbBx8HJAcoBy0HMQc2BzoHPwdDB0gHTQdRB1YHWgdfB2MHaAdtB3EHdgd"
    "7B38HhAeJB40HkgeXB5sHoAelB6kHrgezB7cHvAfBB8YHygfPB9QH2QfdB+IH5wfsB/"
    "EH9Qf6B/"
    "8IBAgJCA0IEggXCBwIIQgmCCsILwg0CDkIPghDCEgITQhSCFcIXAhhCGYIawhwCHUIegh/"
    "CIQIiQiOCJMImAidCKIIpwisCLEItgi7CMAIxQjKCM8I1AjZCN8I5AjpCO4I8wj4CP0JAwkICQ"
    "0JEgkXCR0JIgknCSwJMQk3CTwJQQlGCUwJUQlWCVsJYQlmCWsJcQl2CXsJgQmGCYsJkQmWCZsJ"
    "oQmmCasJsQm2CbwJwQnGCcwJ0QnXCdwJ4gnnCe0J8gn4Cf0KAgoICg0KEwoZCh4KJAopCi8KNA"
    "o6Cj8KRQpKClAKVgpbCmEKZgpsCnIKdwp9CoMKiAqOCpQKmQqfCqUKqgqwCrYKvArBCscKzQrT"
    "CtgK3grkCuoK7wr1CvsLAQsHCwwLEgsYCx4LJAsqCy8LNQs7C0ELRwtNC1MLWQtfC2QLagtwC3"
    "YLfAuCC4gLjguUC5oLoAumC6wLsgu4C74LxAvKC9AL1gvcC+IL6QvvC/"
    "UL+"
    "wwBDAcMDQwTDBkMIAwmDCwMMgw4DD4MRQxLDFEMVwxdDGQMagxwDHYMfQyDDIkMjwyWDJwMogy"
    "oDK8MtQy7DMIMyAzODNUM2wzhDOgM7gz1DPsNAQ0IDQ4NFQ0bDSENKA0uDTUNOw1CDUgNTw1VD"
    "VwNYg1pDW8Ndg18DYMNiQ2QDZYNnQ2kDaoNsQ23Db4NxQ3LDdIN2Q3fDeYN7A3zDfoOAQ4HDg4"
    "OFQ4bDiIOKQ4vDjYOPQ5EDkoOUQ5YDl8OZg5sDnMOeg6BDogOjg6VDpwOow6qDrEOuA6+"
    "DsUOzA7TDtoO4Q7oDu8O9g79DwQPCw8SDxkPIA8nDy4PNQ88D0MPSg9RD1gPXw9mD20PdA97D4"
    "IPiQ+QD5gPnw+mD60PtA+7D8IPyg/RD9gP3w/mD+0P9Q/"
    "8EAMQChASEBkQIBAnEC8QNhA9EEQQTBBTEFoQYhBpEHAQeBB/"
    "EIYQjhCVEJ0QpBCrELMQuhDCEMkQ0BDYEN8Q5xDuEPYQ/"
    "REFEQwRFBEbESMRKhEyETkRQRFIEVARVxFfEWcRbhF2EX0RhRGNEZQRnBGkEasRsxG7EcIRyhH"
    "SEdkR4RHpEfAR+"
    "BIAEggSDxIXEh8SJxIuEjYSPhJGEk4SVRJdEmUSbRJ1En0ShBKMEpQSnBKkEqwStBK8EsQSzBL"
    "UEtsS4xLrEvMS+"
    "xMDEwsTExMbEyMTKxMzEzsTRBNME1QTXBNkE2wTdBN8E4QTjBOUE50TpROtE7UTvRPFE80T1hP"
    "eE+YT7hP2E/"
    "8UBxQPFBcUIBQoFDAUOBRBFEkUURRaFGIUahRzFHsUgxSMFJQUnBSlFK0UthS+"
    "FMYUzxTXFOAU6BTxFPkVARUKFRIVGxUjFSwVNBU9FUUVThVXFV8VaBVwFXkVgRWKFZMVmxWkFa"
    "wVtRW+"
    "FcYVzxXYFeAV6RXyFfoWAxYMFhQWHRYmFi8WNxZAFkkWUhZaFmMWbBZ1Fn4WhhaPFpgWoRaqFr"
    "MWuxbEFs0W1hbfFugW8Rb6FwMXDBcUFx0XJhcvFzgXQRdKF1MXXBdlF24XdxeAF4kXkhecF6UX"
    "rhe3F8AXyRfSF9sX5BftF/"
    "cYABgJGBIYGxgkGC4YNxhAGEkYUhhcGGUYbhh3GIEYihiTGJwYphivGLgYwhjLGNQY3hjnGPAY"
    "+hkDGQwZFhkfGSkZMhk7GUUZThlYGWEZaxl0GX4ZhxmRGZoZpBmtGbcZwBnKGdMZ3RnmGfAZ+"
    "hoDGg0aFhogGioaMxo9GkYaUBpaGmMabRp3GoEaihqUGp4apxqxGrsaxRrOGtga4hrsGvUa/"
    "xsJGxMbHRsnGzAbOhtEG04bWBtiG2wbdRt/"
    "G4kbkxudG6cbsRu7G8UbzxvZG+"
    "Mb7Rv3HAEcCxwVHB8cKRwzHD0cRxxRHFscZRxwHHochByOHJgcohysHLYcwRzLHNUc3xzpHPQc"
    "/h0IHRIdHB0nHTEdOx1FHVAdWh1kHW8deR2DHY4dmB2iHa0dtx3BHcwd1h3hHesd9R4AHgoeFR"
    "4fHioeNB4+"
    "HkkeUx5eHmgecx59Hogekx6dHqgesh69Hsce0h7cHuce8h78HwcfEh8cHycfMh88H0cfUh9cH2"
    "cfch98H4cfkh+dH6cfsh+9H8gf0h/dH+gf8x/+IAggEyAeICkgNCA/"
    "IEogVCBfIGogdSCAIIsgliChIKwgtyDCIM0g2CDjIO4g+"
    "SEEIQ8hGiElITAhOyFGIVEhXCFnIXIhfiGJIZQhnyGqIbUhwCHMIdch4iHtIfgiBCIPIhoiJSI"
    "wIjwiRyJSIl4iaSJ0In8iiyKWIqEirSK4IsMizyLaIuYi8SL8IwgjEyMfIyojNSNBI0wjWCNjI"
    "28jeiOGI5EjnSOoI7QjvyPLI9Yj4iPuI/"
    "kkBSQQJBwkKCQzJD8kSyRWJGIkbiR5JIUkkSScJKgktCS/"
    "JMsk1yTjJO4k+iUGJRIlHiUpJTUlQSVNJVklZSVwJXwliCWUJaAlrCW4JcQl0CXcJecl8yX/"
    "JgsmFyYjJi8mOyZHJlMmXyZrJncmhCaQJpwmqCa0JsAmzCbYJuQm8Cb9JwknFSchJy0nOSdGJ1"
    "InXidqJ3YngyePJ5snpye0J8AnzCfZJ+"
    "Un8Sf9KAooFigjKC8oOyhIKFQoYChtKHkohiiSKJ4oqyi3KMQo0CjdKOko9ikCKQ8pGykoKTQp"
    "QSlNKVopZylzKYApjCmZKaYpsim/Kcwp2CnlKfEp/"
    "ioLKhgqJCoxKj4qSipXKmQqcSp9KooqlyqkKrEqvSrKKtcq5CrxKv4rCisXKyQrMSs+"
    "K0srWCtlK3IrfyuMK5krpSuyK78rzCvZK+"
    "Yr8ywBLA4sGywoLDUsQixPLFwsaSx2LIMskCyeLKssuCzFLNIs3yztLPotBy0ULSEtLy08LUkt"
    "Vi1kLXEtfi2LLZktpi2zLcEtzi3bLekt9i4ELhEuHi4sLjkuRy5ULmEuby58Loouly6lLrIuwC"
    "7NLtsu6C72LwMvES8eLywvOi9HL1UvYi9wL34viy+ZL6cvtC/CL9Av3S/rL/"
    "kwBjAUMCIwLzA9MEswWTBnMHQwgjCQMJ4wrDC5MMcw1TDjMPEw/"
    "zENMRoxKDE2MUQxUjFgMW4xfDGKMZgxpjG0McIx0DHeMewx+"
    "jIIMhYyJDIyMkAyTjJcMmoyeTKHMpUyozKxMr8yzTLcMuoy+"
    "DMGMxQzIzMxMz8zTTNcM2ozeDOGM5UzozOxM8AzzjPcM+sz+"
    "TQHNBY0JDQzNEE0TzReNGw0ezSJNJg0pjS1NMM00jTgNO80/"
    "TUMNRo1KTU3NUY1VDVjNXI1gDWPNZ01rDW7Nck12DXnNfU2BDYTNiE2MDY/"
    "Nk42XDZrNno2iTaXNqY2tTbENtM24TbwNv83DjcdNyw3OzdJN1g3Zzd2N4U3lDejN7I3wTfQN9"
    "837jf9OAw4GzgqODk4SDhXOGY4dTiEOJM4ojixOME40DjfOO44/"
    "TkMORs5Kzk6OUk5WDlnOXc5hjmVOaQ5tDnDOdI54TnxOgA6DzofOi46PTpNOlw6azp7Ooo6mjq"
    "pOrg6yDrXOuc69jsGOxU7JTs0O0Q7UztjO3I7gjuRO6E7sDvAO9A73zvvO/"
    "48DjwePC08PTxNPFw8bDx8PIs8mzyrPLo8yjzaPOo8+"
    "T0JPRk9KT05PUg9WD1oPXg9iD2YPac9tz3HPdc95z33Pgc+Fz4nPjc+Rz5XPmc+dz6HPpc+"
    "pz63Psc+1z7nPvc/Bz8XPyc/Nz9HP1c/Zz94P4g/mD+oP7g/yD/ZP+k/"
    "+UAJQBlAKkA6QEpAWkBrQHtAi0CcQKxAvEDNQN1A7UD+"
    "QQ5BHkEvQT9BT0FgQXBBgUGRQaJBskHDQdNB5EH0QgVCFUImQjZCR0JXQmhCeEKJQppCqkK7Qs"
    "tC3ELtQv1DDkMfQy9DQENRQ2FDckODQ5RDpEO1Q8ZD10PnQ/hECUQaRCtEO0RMRF1EbkR/"
    "RJBEoUSyRMJE00TkRPVFBkUXRShFOUVKRVtFbEV9RY5Fn0WwRcFF0kXjRfRGBUYXRihGOUZKRl"
    "tGbEZ9Ro9GoEaxRsJG00bkRvZHB0cYRylHO0dMR11HbkeAR5FHoke0R8VH1kfoR/"
    "lICkgcSC1IP0hQSGFIc0iESJZIp0i5SMpI3EjtSP9JEEkiSTNJRUlWSWhJekmLSZ1JrknASdJJ"
    "40n1SgZKGEoqSjtKTUpfSnFKgkqUSqZKt0rJSttK7Ur/"
    "SxBLIks0S0ZLWEtpS3tLjUufS7FLw0vVS+dL+"
    "UwKTBxMLkxATFJMZEx2TIhMmkysTL5M0EziTPRNBk0ZTStNPU1PTWFNc02FTZdNqU28Tc5N4E3"
    "yTgROF04pTjtOTU5fTnJOhE6WTqlOu07NTt9O8k8ETxZPKU87T05PYE9yT4VPl0+qT7xPzk/"
    "hT/NQBlAYUCtQPVBQUGJQdVCHUJpQrVC/"
    "UNJQ5FD3UQlRHFEvUUFRVFFnUXlRjFGfUbFRxFHXUelR/"
    "FIPUiJSNFJHUlpSbVKAUpJSpVK4UstS3lLxUwRTFlMpUzxTT1NiU3VTiFObU65TwVPUU+dT+"
    "lQNVCBUM1RGVFlUbFR/VJJUpVS4VMtU3lTyVQVVGFUrVT5VUVVlVXhVi1WeVbFVxVXYVetV/"
    "lYSViVWOFZLVl9WclaFVplWrFa/VtNW5lb6Vw1XIFc0V0dXW1duV4JXlVepV7xX0FfjV/"
    "dYClgeWDFYRVhYWGxYgFiTWKdYuljOWOJY9VkJWR1ZMFlEWVhZa1l/"
    "WZNZp1m6Wc5Z4ln2WglaHVoxWkVaWVpsWoBalFqoWrxa0FrkWvhbC1sfWzNbR1tbW29bg1uXW6"
    "tbv1vTW+db+"
    "1wPXCNcN1xLXGBcdFyIXJxcsFzEXNhc7F0BXRVdKV09XVFdZV16XY5dol22Xctd313zXgheHF4"
    "wXkReWV5tXoJell6qXr9e017nXvxfEF8lXzlfTl9iX3dfi1+gX7RfyV/dX/"
    "JgBmAbYC9gRGBYYG1ggmCWYKtgv2DUYOlg/"
    "WESYSdhO2FQYWVhemGOYaNhuGHNYeFh9mILYiBiNWJJYl5ic2KIYp1ismLHYtti8GMFYxpjL2N"
    "EY1ljbmODY5hjrWPCY9dj7GQBZBZkK2RAZFVkamR/ZJVkqmS/"
    "ZNRk6WT+ZRNlKWU+"
    "ZVNlaGV9ZZNlqGW9ZdJl6GX9ZhJmJ2Y9ZlJmZ2Z9ZpJmp2a9ZtJm6Gb9ZxJnKGc9Z1NnaGd+"
    "Z5NnqWe+Z9Rn6Wf/aBRoKmg/"
    "aFVoamiAaJZoq2jBaNZo7GkCaRdpLWlDaVhpbmmEaZlpr2nFadtp8GoGahxqMmpIal1qc2qJap"
    "9qtWrKauBq9msMayJrOGtOa2RremuQa6ZrvGvSa+hr/"
    "mwUbCpsQGxWbGxsgmyYbK5sxGzabPBtBm0cbTNtSW1fbXVti22hbbhtzm3kbfpuEW4nbj1uU25"
    "qboBulm6tbsNu2W7wbwZvHG8zb0lvYG92b4xvo2+5b9Bv5m/"
    "9cBNwKnBAcFdwbXCEcJpwsXDHcN5w9HELcSJxOHFPcWZxfHGTcapxwHHXce5yBHIbcjJySHJfc"
    "nZyjXKkcrpy0XLocv9zFnMsc0NzWnNxc4hzn3O2c81z5HP6dBF0KHQ/"
    "dFZ0bXSEdJt0snTJdOB093UOdSZ1PXVUdWt1gnWZdbB1x3XedfZ2DXYkdjt2UnZqdoF2mHavds"
    "d23nb1dwx3JHc7d1J3aneBd5h3sHfHd9539ngNeCV4PHhUeGt4gniaeLF4yXjgePh5D3kneT55"
    "VnlueYV5nXm0ecx543n7ehN6KnpCelp6cXqJeqF6uHrQeuh7AHsXey97R3tfe3Z7jnume7571n"
    "vufAV8HXw1fE18ZXx9fJV8rXzFfNx89H0MfSR9PH1UfWx9hH2cfbR9zX3lff1+FX4tfkV+"
    "XX51fo1+pX6+ftZ+7n8Gfx5/N39Pf2d/f3+Xf7B/yH/gf/"
    "mAEYApgEGAWoBygIqAo4C7gNSA7IEEgR2BNYFOgWaBf4GXgbCByIHhgfmCEoIqgkOCW4J0goyC"
    "pYK+gtaC74MHgyCDOYNRg2qDg4Obg7SDzYPlg/"
    "6EF4QwhEiEYYR6hJOErITEhN2E9oUPhSiFQYVahXKFi4Wkhb2F1oXvhgiGIYY6hlOGbIaFhp6G"
    "t4bQhumHAocbhzSHTYdnh4CHmYeyh8uH5If9iBeIMIhJiGKIe4iViK6Ix4jgiPqJE4ksiUaJX4"
    "l4iZGJq4nEid6J94oQiiqKQ4pdinaKj4qpisKK3Ir1iw+LKItCi1uLdYuOi6iLwovbi/"
    "WMDowojEKMW4x1jI+"
    "MqIzCjNyM9Y0PjSmNQo1cjXaNkI2pjcON3Y33jhGOK45Ejl6OeI6SjqyOxo7gjvqPE48tj0ePY"
    "Y97j5WPr4/Jj+OP/"
    "ZAXkDGQS5BlkH+"
    "QmpC0kM6Q6JECkRyRNpFQkWuRhZGfkbmR05HukgiSIpI8kleScZKLkqaSwJLakvSTD5Mpk0STX"
    "pN4k5OTrZPIk+KT/"
    "JQXlDGUTJRmlIGUm5S2lNCU65UFlSCVO5VVlXCVipWllcCV2pX1lg+WKpZFll+"
    "WepaVlrCWypbllwCXG5c1l1CXa5eGl6GXu5fWl/GYDJgnmEKYXZh3mJKYrZjImOOY/"
    "pkZmTSZT5lqmYWZoJm7mdaZ8ZoMmieaQppemnmalJqvmsqa5ZsAmxybN5tSm22biJukm7+"
    "b2pv1nBGcLJxHnGOcfpyZnLWc0JzrnQedIp09nVmddJ2Qnaudxp3inf2eGZ40nlCea56HnqKev"
    "p7anvWfEZ8sn0ifY59/n5uftp/"
    "Sn+6gCaAloEGgXKB4oJSgsKDLoOehA6EfoTqhVqFyoY6hqqHGoeGh/"
    "aIZojWiUaJtoomipaLBot2i+"
    "aMVozGjTaNpo4WjoaO9o9mj9aQRpC2kSaRlpIGknqS6pNak8qUOpSqlR6VjpX+"
    "lm6W4pdSl8KYMpimmRaZhpn6mmqa2ptOm76cLpyinRKdgp32nmae2p9Kn76gLqCioRKhhqH2om"
    "qi2qNOo76kMqSmpRaliqX6pm6m4qdSp8aoOqiqqR6pkqoCqnaq6qteq86sQqy2rSqtnq4OroKu"
    "9q9qr96wUrDCsTaxqrIespKzBrN6s+"
    "60YrTWtUq1vrYytqa3GreOuAK4drjquV650rpKur67MrumvBq8jr0CvXq97r5ivta/Tr/"
    "CwDbAqsEiwZbCCsJ+"
    "wvbDasPexFbEysVCxbbGKsaixxbHjsgCyHrI7slmydrKUsrGyz7LsswqzJ7NFs2KzgLOes7uz2"
    "bP2tBS0MrRPtG20i7SotMa05LUCtR+1PbVbtXm1lrW0tdK18LYOtiy2SbZntoW2o7bBtt+2/"
    "bcbtzm3V7d1t5O3sbfPt+"
    "24C7gpuEe4ZbiDuKG4v7jduPu5Gbk4uVa5dLmSubC5zrntugu6KbpHuma6hLqiusC637r9uxu7"
    "OrtYu3a7lbuzu9G78LwOvC28S7xqvIi8przFvOO9Ar0gvT+9Xb18vZu9ub3Yvfa+Fb4zvlK+"
    "cb6Pvq6+zb7rvwq/Kb9Hv2a/hb+kv8K/4cAAwB/"
    "APsBcwHvAmsC5wNjA98EVwTTBU8FywZHBsMHPwe7CDcIswkvCasKJwqjCx8LmwwXDJMNDw2LDg"
    "cOgw8DD38P+"
    "xB3EPMRbxHvEmsS5xNjE98UXxTbFVcV1xZTFs8XSxfLGEcYwxlDGb8aPxq7GzcbtxwzHLMdLx2"
    "vHiseqx8nH6cgIyCjIR8hnyIbIpsjFyOXJBckkyUTJZMmDyaPJw8niygLKIspBymHKgcqhysDK"
    "4MsAyyDLQMtfy3/Ln8u/y9/L/8wfzD/MXsx+zJ7MvszezP7NHs0+zV7Nfs2ezb7N3s3+zh/"
    "OP85fzn/On86/zt/O/88gz0DPYM+Az6DPwc/"
    "h0AHQIdBC0GLQgtCi0MPQ49ED0STRRNFl0YXRpdHG0ebSB9In0kfSaNKI0qnSydLq0wrTK9NM0"
    "2zTjdOt087T7tQP1DDUUNRx1JLUstTT1PTVFNU11VbVd9WX1bjV2dX61hrWO9Zc1n3Wnta/1t/"
    "XANch10LXY9eE16XXxtfn2AjYKdhK2GvYjNit2M7Y79kQ2THZUtlz2ZTZtdnW2fjaGdo62lvaf"
    "Nqe2r/a4NsB2yLbRNtl24bbqNvJ2+rcC9wt3E7cb9yR3LLc1Nz13RbdON1Z3XvdnN2+3d/"
    "eAd4i3kTeZd6H3qjeyt7s3w3fL99Q33LflN+139ff+eAa4DzgXuB/"
    "4KHgw+Dl4QbhKOFK4WzhjeGv4dHh8+"
    "IV4jfiWeJ64pzivuLg4wLjJONG42jjiuOs487j8OQS5DTkVuR45JrkvOTe5QHlI+"
    "VF5WflieWr5c3l8OYS5jTmVuZ55pvmvebf5wLnJOdG52nni+"
    "et59Dn8ugU6DfoWeh76J7owOjj6QXpKOlK6W3pj+my6dTp9+"
    "oZ6jzqXuqB6qTqxurp6wvrLutR63Prluu569zr/"
    "uwh7ETsZuyJ7Kzsz+zy7RTtN+1a7X3toO3D7eXuCO4r7k7uce6U7rfu2u797yDvQ+9m74nvrO/"
    "P7/LwFfA48FvwfvCh8MXw6PEL8S7xUfF08Zjxu/He8gHyJPJI8mvyjvKx8tXy+PMb8z/"
    "zYvOF86nzzPPw9BP0NvRa9H30ofTE9Oj1C/Uv9VL1dvWZ9b314PYE9if2S/"
    "Zv9pL2tvbZ9v33IfdE92j3jPew99P39/"
    "gb+D74YviG+Kr4zvjx+RX5Ofld+YH5pfnJ+ez6EPo0+lj6fPqg+sT66PsM+zD7VPt4+"
    "5z7wPvk/Aj8LPxQ/HX8mfy9/OH9Bf0p/U39cv2W/br93v4C/if+S/5v/pT+uP7c/wD/Jf9J/"
    "23/kv+2/9v//2Nocm0AAAAAAAMAAAAAo9cAAFR8AABMzQAAmZoAACZnAAAPXP/"
    "bAEMABgQFBgUEBgYFBgcHBggKEAoKCQkKFA4PDBAXFBgYFxQWFhodJR8aGyMcFhYgLCAjJicpK"
    "ikZHy0wLSgwJSgpKP/"
    "bAEMBBwcHCggKEwoKEygaFhooKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoK"
    "CgoKCgoKCgoKCgoKP/AABEIAgACAAMBIgACEQEDEQH/"
    "xAAcAAABBQEBAQAAAAAAAAAAAAABAAIDBAUGBwj/"
    "xABNEAABAwIEAwMJBQIMBgEEAwABAAIRAwQFEiExQVFhBhMiFDJxgZGhscHRI0JSYuEHchUkM0"
    "NzgpKTssLS8BY0U2Oi8UQlJnSDNVRk/8QAGgEAAwEBAQEAAAAAAAAAAAAAAAECAwQFBv/"
    "EACERAQEBAQEAAgMBAQEBAAAAAAABEQIDEiEEEzFBFCJR/"
    "9oADAMBAAIRAxEAPwDxUiFGSpXBRuCT20Z84FdP2Qqhta3B/wD7Lfi1cw5a/"
    "ZqtkvLUEgfxhp1PULP1n0y6e22Tg6i6PxKYqlgrxUtXkEHxkaHoFfyrxPf+"
    "ufo0BPAQywU4Ll1jTgEYSaiikAToQCeEYEZamERqrIbKZWbDSUSDnpG14bBMrle29w3+"
    "DL0QZ8H+Jql7S4o6zoXIaQCzLuRxI6dV5/"
    "iWKuuzUFRzYdEiRwhdflza0nSx2QpC5xuuCSB3BPvaum7Sdm82G0ajamrqgOrubT0XD4biww67"
    "fUpZZLMvnDmDx9C3brtVXu7SlSL2gNg7tPCOXVdM56TfSRl0cYqYLcPGRr8k0dp2PpHJUsU7UV"
    "L5mQ0mAB+bzT169VLUpULmq99eow5iXGXRqfWqF+"
    "20pU4pFkh0aPnn1W3PNZ301SzvquLhll2qrXDqgBnJE8JV3D7wU7l8RGUga9Qu+"
    "sqFhe2VAVO7zGm15mqd49PVazlneteT1KpkyBumtq67Lb7T4eKFau6k0ZO/"
    "cGwSdJKwGtgxxWk5ZLTHEhW26yqTJjRSBzhz9iuQLLqGbiqtWgWxqFbpCq+coJjohVbU0zA+"
    "xAZTmnRDJCum2efun2FTW9EeLMEvkVVbZvnepdDYvrsa4U6rmjTQOIVQ2Jd5jfirdlQr0qniY7"
    "KSJ8JWfdipHQWNzctLftCfH+IrpMPxC5aBD/v8z06rGwx8gZmu8/"
    "l6F0Vm6noIM5lw+mKkjTt8UutPEN+Z+q1rG9r1HNzO3eBuVlU8vDmtCynvGEA+"
    "eFzdSNOXUWPeOiSPO5rTNbuKbi+"
    "TALtFmWNTJTJcYh0rB7ZYr3ba1NlSC62PAfmWc51tHV08WpVHNaGvkmNh9Vda/"
    "OJErwfCsRrfwjaTUMd8zgOYXs3Zqqa9i9xMxUI9wT68ry0laR3QlOeIJTCs8UUpJJJAkCigkVA"
    "7JpTidFXr1Qzc8URFqYmFVrXQZMu4xsVQubs6hpPncgsy5L6gO+"
    "pnZaSMr00Lq7puB8QnNyKz6tcSYPHks2q14J0O/JRjNOoPsTxh12u1K54OO/VVatd/"
    "4jvzKBmNj7FBUDuR35InLP5j3zi4yVq2FYhh14BYYY4uPhPsWxbUXU2Qd4ATwvnTb28d4PFz5r"
    "KNVx+8Ubp05fWoAUYPnWpb3L25vF8VOapdudll03gSp21wJ2QXzqw+nMLRstM/"
    "qWMbodFPaXzaeaRvCvIWm4m1wDD6fksyYXVXVJl1ScKbtWtPHmsC6sqlF2rSdJ2KA8OdwTHKSo"
    "IhQkr3dfRmOU+GOLL62O0VWn3hQlOt3ZbikeTgfep7+"
    "4ix7T2Qr5rF2pP2xHuauliVwHYW5zW7Wxvcx7mr0MNkFeN+"
    "RMrk9LiIhNOhUpCieIK42FpzSiXBQzqmPfunmlqyHap4cANVmVq/"
    "dsLo2XO432g8lZWaKcluX7vo6rbny0tdm+7o0Wlz3QBvoVlYl2mw+"
    "3p1Gvf4mx913TovKca7YXj++p0nOY3wxBcI2/"
    "MuTvcYvrmq9z7ipDokZ3fVdHn4F8noHa3G7a+"
    "F0KDpz5I34RzHRcW9xLiZMelZ9K7qFozve48ZMpz7vQ6Fd/"
    "HnJCvdTVSMx11U1Br3RD3bc1ntq53mZWjQqANbEzCv4Mb0NenWyCHu35rMrU62Y5i4681quqmF"
    "C8tdu0b8lU5hazA91MzrO260bDGKtF8F9SA2P5QqpcMGsc1BTpy8pyQ9bGIYiLqg1pknMHGSTw"
    "KyHMJcSOJUhYQ0apU3gOhwVFprXZRqnd4FaNsx3mgqCrblsQD7U1RrYUQ7vdOXzVrEKMd3DRx+"
    "Sq4O0/"
    "bT0+a37ikyplkDRKrnLDo1qb58I06K1b4camaI06BN8jbS2B19C6jB7ZtXvvDMRy6rm9OsV+"
    "uKlrhbxm0B24D6rZZg9QnSm32D6robTDGeOWcuS3KVpSaTNNvsC4/"
    "T1HxxyVnhFQRLGjxch9VsUMJOhIAg8B+q220qY2Y0epTMaI2C5OvQ5yzaOHQRJ4/"
    "74rUsrAAt55k5jRmGnFaVm0S0x95T87VyG3FB9C1rP4NYXewLzTtdfGreNbO9GPe5ekdor9lvY"
    "3TMwBNu8jQ8ivHcVuPKLlj5mGge8rt8PPfuritZnJe27uVRp969l7A3PfWL2zvWd/"
    "hC8XktcHD7uq9G/ZdiMvbSc8eKs8xB/AFr78/"
    "S49KqtIJUTlO94cCQq7zuvP6mHoJSmFyBdooLT8yE6qLMZSzdUqNSPPhKzb0OLjvEjh0V8u0UN"
    "RocNp1RGfTIe0DUjiq73t209q1KtEOkRxVarZtaM0HU9FpGXTOe1jt/"
    "ih3FHclo9ahvqndSGnZ0LKrXTzMOO/"
    "MqpGHTXqCg3i3ePOUJdQJI8P8AaWM6o87uPtTM7h94+"
    "1VGborehQqO3btO6069BlPLl4rk7W5ex3nHbqujZc97MmY9KYc1ckeH1qEKe5pkZd+"
    "KrhGEkBRlNlIpYRpJTmHdNKQMJ4GvZXrmP1Igkclr0+5u2HPlk+"
    "Hzv981zFGS4DqFrWbXMAdJADpSqngNYaBV4Vq6EAetVl7r6M2NEwaPaeRUsJjhCP8AE9O47B3I"
    "a+gwne7by/KvYaMOYSNpXgnZK4FO9tBJ/"
    "wCZYd+"
    "oXumD1RUtXnfxke4LyvyuP9cfoncxQVWGfUrbnBV6zwJ9C86uZTqAgKB5iZU1WqDoqtYlzSAq5"
    "hVj49fNo29dod4hl2jmF5vjtV9xWruEkOy8OgXoOL2Dqwqu0MxuPQuWvcLeXvaAOHAfVdXlYnX"
    "GPtS4Elu/"
    "pWZeUzTe8coXcXGGvo2znEDToOa5HFmRd1Wxy+"
    "AXoeeFrMkwmudwVxls5zA4BQ1rZwcdFt9JRMcZ0UzXP6+"
    "xPtbYvdEcFqsw0mm0xuP98VVsTYymVXg7n2Kwxzj7OSndh7g46cf98VLTsagiBwUWwsVxTc/"
    "h1VuztCX+"
    "Jv3eqt21lUkS3hzCvUbSo06N4cws76SHiGth2agzK0TpxPJYtSwqCs8ZOJ5812NCjVeA0N2HMJ"
    "/8EVXEuyb67j6pfug+LnaGG1fF4PirNTDHiJYPeuxtcMb4s4jb/e6jxC1bT7uOM/"
    "JP9ka881yNlQdTz6RMLVhMawBOzELT7ro54Nq0XuiGO9i0sCvW0O/"
    "zkjNl5dV0dpg1K4z5Wk5Y2BPzXGVmm0jMCM3PTZc/r9rnD2Czq06ufLwhTkjmuawLEBV7/"
    "Ll0y7OB5rfbmcePsXl+8qeuMTCFIwJtKiTvO/"
    "JWWUT135LlxOBSb4h6VfYRTtqjnaZQT7kyjQ2Ou/JZ/aK+NlbXDGxJoOdqRyP0W/"
    "lzdDle2l82pc5W5SDbx73LiJ8YbxKvYjfPu6wc8/"
    "dy8Oqo0xnvqDeBc0e9e14cZFSNSzw2pcWNaqGuhsjY8p5K92QqixxuhTcRILiZ/"
    "cK7HsrhNOt2fvCQZzvGx/AOq4fFKRw3tPVySBTjcRuwc/"
    "Sl7Rp8Xs2HXArWNN4I1nb0lSucDK5zsjeur4TaZiPFn4j8TluvdBK8v0/"
    "qaLnJpcon1IJTQ+eKx1NqQuTS/"
    "VAkRuoaj4JQnVgOlSAgAFUG14fBj2q5IfTbHKVUhXpn4ld92zQffjUelZYvTUcQWjnsp8TpEl2"
    "/nnh6Vlmi8EkB3sWkjPqn3rA9s83SsesIcR1W3RaSYdOgSrWVNzZ4kyrkYdVz6BC2RhrS4xPsK"
    "d5Gynq7bbXRDJjMC17F85/Un/xduhLJH51G9jdO7I6xqnhp7u1zZIPNYr2FsbK/"
    "TvHicysVKTKsRGnIynhYxJKSsV6Bbl0OvRMazdBYAZKORWGtUtKjmnZKiw2zokPknYgq/"
    "VqhlNwjcFS3FMUWPgR4SVi3FaXb8Eh/HjuIsyBnrVELYxlh7xo4Bk/"
    "79iyC0he0+"
    "lBMeESYTSZVxFWsJrGjf2pBiKzT7wvbux9932HvJdP2xHHk1eDUzluKZ5OB969L7DXZFpEn/"
    "mOQ5NXF+Vxsc3p/"
    "HptR3I8FTuHmd+"
    "Cqm8giSfYEypcZ5ido2Xk9c5XFeppEy5IiVXdUIOspprgcUROn3DMzHCJVB1k19SSzf0K4K4Ji"
    "Vbti12X1qubZUubxXDmm0qtDNdOXMLz7GcKi8rHIeHEcgvbX27XtMga9Vi4lhDa1SoQwGY4nou"
    "nj1sGvK7LDgabAWc+SNzhAOYiny5L0mjgmRw8DRHVynOFtA8TW+"
    "0rT99GvKKOHOpvJycI3C3LKya9rWuZs0cl278LpuiPDHJ31T6eFNbqHnbn+im+9Djf4HYTPd7+"
    "hXbfBqJAmnw/KuqNoGtHiPLdGlQAdudlN96JGba4BRLWnut2j8P0U/"
    "wDANIbUv8K36DgxrejYT3Vgsr7KxzrMIbTcSKfTgrTLJoaB3bduQWo6qChnlR+"
    "y6qcsk2pGzY9iwsbOTudd83yXZOcAvPMZuO97mBtPyXR5dXutZGdTGaZUlVsRCdTbEoVXbL1JM"
    "jfl6X2Lc2t5Zm1jJ/mXK9t8K7jyLLTic/EflXR9gP8A58/9v/Mul7c4YKvkXiIjPx/"
    "d6LHvnWsmvKOyt66j5VLt8u89V6taM7wB0feheLW48nzccy9t7PuFe0c7aKhHuC4fThl3FxlMD"
    "hxUzQOSc5uUph3C5LyylTuc2nQqO2ygn3Lzntzic3eRrxDreNjzcuzxav3VhdaT9i4+"
    "4ryPHavlF6x8RDAPeVr5T7NRaczSVYw9odf2079634hQAZWFWsHbmxG0/pmD3hetxPppzHs/"
    "Y5n/ANIrtjeq7/C1ee/tDt+5xfEKumnd7D8rQvTOydLJhtUT/On4BcR+0u3zvxJ0793/AJVn6/"
    "xt8fof2f3bTZ2FMkz9pz5uXb1HgtJC8s7EuNK6smbxn+Dl6VSfNAf74rzfSMevo2s6JVc18vE+"
    "1S1tQVTqN1K52NWG3APP2oPrAgqsBooa5IY5NFqStcMY8ydk9mMUqYA10Ebn6LEuSSXbcFRew6"
    "+lVE2t+"
    "tiDa7nBs7zuVcpZHU2kgagHZc5YiapH5VuVXGna0iOQHuWvLLqq13cU6ZdH4o0VZt6HEgT7VlX"
    "dw99xVBJgPPHqrdrTENceLVpjO1u2dQOgni2VFietEZfx/"
    "VZjr00tGTponeWmq0B080YTEu3VBVfDneceK0LGqTnzE8NypHWrKzi6BzRFm6n5rhr1TwYN1RP"
    "hy9VFaXfd5886xxWiKRf50aLJuqHd5YO8owY2n0W14yxp0WXUplkTxVjDa1Q95JnbeequX1qBk"
    "9aMDKzQpKdbLO+"
    "qrFxTcyjE10PeNuqT43ILVlXVhUa6QNAJ4KGhcvpkQ7SZ4rRo3rXiKmUyY2OyMJ5FilIOeT+"
    "T6rDrsDZ9C6K8Ac1x/KVg3jdHfur2X0rNedUAmvOqQKqVFPb/ACjT1C7nsZXayi1p43A/"
    "yrhW7hdH2brOY6kAY+2HyWPtNjl9o9PqVA4y1S27m5fFG/"
    "NZdtULm6nirAcQvL75+"
    "3md2yrdwWkuiPaqb9yjmJ3KSj4p01s5grtvVLMvSVU4SiCYU2D5NZt3oBp7U/"
    "v2uEkj2rGDjO5Vmi1zo39qIcq3VuA2Yj2qu6uXEp3kzncDr1Ce2zI+6faErVyIWhzj79lM2m/"
    "r7EQw0jJHRPFzGmmnRLVYidSceB9ieygeR25KzQqh7tQ3bktCjRDwIA2nZKqkZfdOgaH2IGk7k"
    "fYuhZaCBIG3JCpZiNAN+SWKkc6aZ6+xIN5rafZ9B7FC+"
    "zM7D2IxpIyLgZcq82vXZskcJXqmJ27m93pz+S8pqiIldv40+"
    "1pG8VFWbMKcBNqNJhepf405jv8A9n8ny+f+3/mXqGKUBcd1IJyzt6l5T2Aqx5fv/"
    "N8f3l7DA4rLp0cfx84doLI2/"
    "caOGbNuD0XonYa7LrEgka3Eb9GrI7e2Bb5Dla0fymwH5VT7E3xa2nT18VyPvfurn9Ofpl6R6jW"
    "cJ9SgzahIS9pKY5pFNx5ArkvP253P9rbo06dZgjW3dx/"
    "eXmj3Go8Ers+2dxFdzSTJt+fVy4im6R610eXH2P8AUjhoVf7PsBxC0/p2fELOcdCtbsy3NiFp/"
    "wDkM+IXb/I6fOPbOzrctk/+kPwC53t9bNdZX1SNfs/"
    "i1dRgbYtXAfjPwCxO3TIwi+P7n+Jqx7+436/"
    "jzLAn93jlBu0Zv8JXo9m7NaMPp+"
    "K8ww58do6Y9P8AgK9LwszYUvX8SuL05cvos7hMcyZUgRXJZ9sKrmkY0Cp3bSGv05LWEJtegx9J"
    "x0k/"
    "lTkZ9RzFXcquQCVexKmKdZwERI4dFnPcR7Vc5Z1dsgxtQkx5vNaFcirQY1hBIg+"
    "5c46u5p0J9qv4Vcl9YhxJ8HE+haSM+"
    "kFWyc2q95boXHmoqtTu25QYgwulvKbfJaboGpHDouRvnRWqDk8q2aN75cT1SFSOKilKU8EWqd0"
    "5o3PsC1LbEmDNnceHJYIRdpsiKj0SwfRrd5BBiOPpVTE8KNTu8jBpM6nos7AK1Q9//"
    "V+a626qNp5Z4ynh2OZoUWW2bNlGbqquIXrHd3ldtPLoquIX7nd3Gm/"
    "y6LLdULtykmiXJSo8yOYKWfQklJpKYXJzXaJnHDViCx2vBY983wu/"
    "cK0jmLTIO3JULoSDI+6vXj6VgVGkOCTd1YuKfi05KvEFXE1K0rUwaoG16I/"
    "7rfiFkhWrB5bcUv6QfELP0n0x7mx6lhlQPokyPO+QV4bLDwCqXWxn/"
    "qfILaaV5vpMeX685TkkklgwKdEpQRa0k7FTRgg6rRsG5jT9fzVahbl2XQ69Ft4ZbQaeh48PSku"
    "RPQpeBuqstpabq1TogNBT8oCmtZGZVtM/HjO6jZhwzHxH2/"
    "otUgJaKVKlKxa2PEduav0aIaBqdoSpxPqU7SEKkFrUS1DOmuqDmhQOZKY5gCJqjmgagPFEXGfi"
    "rP5L1/JeM3bMuT1r2nEnD7P1/JePYjH2fr+S7fx/"
    "6aMBNeNk5M1XpX+NeK7XsPDfLf6n+Zewt1Xi3ZCpl8r1/B/"
    "mXtNu4HMsbXTw4btzaNqeRTBjPv8A1V5v2Wq91eWrdYNw0+8L1rt617MO72no+"
    "nSquaeRABC8RtXmjiVqAY+0afes+oz949wtagdQcQDufgi538Wqkj7p+"
    "C53AboutXFzv5w8uQV+6vGsoVBm+4eXJc1/"
    "rk1w3beqTiDANjQHxcuZpDwn0rb7UVhWvWOaZ+yA95WMwQ0rr8YcpHYrb7KNm+tP/"
    "wAlnxCw3FdN2KoOqXNq4Da6aPe1b9Onze04K0Cif3z8Asft2B/At9/U/"
    "wATVu4awsYQfxE+5c3+0B2XB7/WP5P/ABNWHX8b9fx5TYj/"
    "AO52ev8AwL0vCv8A+Ppev4leb4QM/aCkd5n/"
    "AAFek4f4bKmPT8SuT0cfpVlNqOysJTgVDcQWOC5L/"
    "WOq1a9yOIBGnQqtUxCoZAdp60qluXVCYEFRVaORp02T5jPqqtzVdVcS4yVVqfNWKggqvWMj1rW"
    "RjVKtxjmrGF1Ay4dJ+78wq1bj6UyicryeivEO5ytrWdEDXwtPuXJYpQyVqpj+"
    "cPxK3sIr5gxs7Ux8lXxi3c5hc1szUnj1QHMO0QBUlamWvdIjUhRFNOHSnAFyaFYtqebN6kQ+"
    "a7Ds9atHlGn4eXVX8aqlvcwd83ySwdmTvusfNUsdqA9x/"
    "W+SvVVx9y4uy681DKVU7KMlQinZkwu6ppcmOKVZ0/"
    "OUQ8qGU5pJICBGA6k3KQANRyWXe2xExHm8lstbm1Ta1KQdeC9d9Jrj7iic23BVH0iOHBdVcWoM"
    "68FlXlvlnX7sqgxSIUtoYr0/"
    "3x8UarYPqUbTDgeRS6R09D7NVAbcidTV+"
    "QXSNaYXn3Zi8y1KLCN644eheq4NRF1TJkjxEe4LzvaOD24ZZBBRaxxOi6R+GjXxH2/"
    "onUcPa0jxH2/"
    "ouSuT4MOhaPqFsDf0LStsNdDSQOPALVp0xSaNzCFS5ayfCZHRSeI6Fk1mWcun5VepMZTA206LM"
    "qX0EwCky8Lo3RDbPeNhMdUCzmXBMbqQVp5pVUq0XpSSq3felHygDgVOK1aBIR76OJVQ3II0BUT"
    "687SjBq465A3JUZuWniVmVqrtYJ3QY9x4lB/"
    "JpG4HMoi4Cz854pZ+qeD5LtzUFXLHCV5Jiwy916/"
    "kvUQ7qvNMcAHcf1vkunw+qqdKxSaOiAMp0r0rfpvx02Oz1fu/KN9cvH0r3LDn5+86R818/"
    "YZW7vvdYmPmveMFqip30GYy/NZV1+Kn23bnwisP+xV/wrwi6aKWJW5PAtPvXvXbRp/gi4j/"
    "AKFX/CvBMRny+jPIfEp59F+Q7fs/dNNk/wDpD8ArF/"
    "cfZVNfuFYeCVC22cAfvn4BWr6o7uamv3D8Fx9T/wBPO6v257FX5q4/"
    "c+ZVIOgJ148uqiTwUS7fKZFcCTLgu4/Z9Tk0D/8A62/5Vww88TzXoX7OwC2lp/"
    "8ALH+VV07PN65bCBPVcN+0q5LMNxFnLu/i1dw53dUXPOgBXk37TsTDjiVEPE/"
    "ZaQfyFYdNLfpgdl2ipits/nm/wlejW4y0Gj/e6827GOm4s3n8/"
    "wAHL0eg8Gi3Vcvbi9btPqPygqpWrkSrTxIKq1qUzoua/"
    "wBYaVGoHxKbcgZXepNYwthKpJaVfJVnV2+"
    "IqrUb8Vdqt1PpVWp81pGVUa7NJ6qtsSrlc6etVXI1LSwu7LKkGNGRv6F0rWMuLenMGQHaehcOx"
    "xaZGi38IvnB4a52gpxx6JlVbFrIsL3AO1qHgeqxHtLXEGd13z2U7mk2QDMO2XOYrYZC5zW71Dy"
    "6oEYkq/Y6556Ks6idNFZs5bn9Sasd1hzwO824cfSsbG6v8j/"
    "W+SvWdQDPryWXioLu6jWJ+"
    "SLRXMPOyiKmewiJCZkKWosRFDKSrApnknd10S1FisGKa2o53s0PnAbKVtMzstLDLYuewwP5QfJ"
    "EqpHGWcGmZ/"
    "EpajQZjkqtqTkPpVunr7V7Uj37cVKtLfTgs+7oZgdPureNMEHb2KpXpgTttyVYXycld2pafN+"
    "71VF1GBsuou6YM6DzeSxrmllO3BKwapWjzQuKTpgNeHe9eqdgcYpvim6pq6q7QwPuBeV1B8Fdw"
    "PEXWN5TfmeACTo4jdsLk9eNY+"
    "k19BGuwszBwI9KrVr1jQRm19S4zBO0LLihRplz8zs27ieJ6LVdWFQFwJgri9OLHF6TGhXvcxcA"
    "73BQZy86ndUHP1T2VYAWOOa9LbhpKiNTKd9kzvtFFUfMwhPyWBckO3+"
    "ClF5p5x9gWckCjFTpoOuidnH2BRPuHHifYFWzFKUsHyWBXdz9yeys4nf3Ku0Sp2Ujv0RYfyWqZ"
    "zRJ4KQuDRuFWbmao6tV23VI/"
    "ktGqOaYaw5ql3hRBJTwfJoU62+q4vHrZ57jT8XPouuoUy7MqOM2WbuYB0nYTyWvn/"
    "V81xIKdKDmlu6QXbOtjp4p9LivduyX/wAufyfNeEt8K9n7EXwreWyW6ZPvD8yrHZ59Rsds4/"
    "gi4/oKv+FfP2M+G/pfuj4lfQHbMZcHuDP8xV/wr58xo5r2mfyD4lP/AAet2NrAqk0dD/"
    "OfRal9/IVf3D8Fidn5FIf0v0W1fH7Cr+4fguLu/bzu/pylz/Kj0KMKS51qD0JgXX5/w/"
    "OiwZqrGjckBenfs3s3eTscZ0uwfc1eaWw/jtAc3t+K9n/"
    "Z3RAw5xja5+TVVrv866HtNdNtsJuSY8OWZ/"
    "eC8B7WXXlnaK6DYh+TbowfRexftDuhSwrEGZmgju+"
    "P5mrxNzfKMdLt83L91ZdD0uR0XZqmaNtbO5ZviV11lXkMGnFc5hlPu7SmOU/"
    "ErasTBZrzXL6Rwdd/"
    "bbY+"
    "WhEnRVWVIaNUXVoHBY4i1M4hQVXBoOqjfXhpMj2qjdXZAdqOHFVINOr1QZjmqFWpqfSoalwST6"
    "VC55KuRnT6rpHrUDinGSEBTcTsfYkRkq1bPyunomMtnOOzvYp6dq8cHbfhRox0OF1Scun3B8ld"
    "uKYq0wDI1lZFi11OJnzY2V4VDG6WjGdUs8seL3qEW+"
    "XitYmd1GQEzwyi8tnZSvh8TwTNAml3VIK1S0BjU+1RG0HM+"
    "1XXFMlIrFQWY5n2p4tBzPtVpFKpsV22okalX7QCk2NT4pVZzwEBVHNLTx5vZiWf1ldpiB61Qw0"
    "yzX8f0Wq1vhK+h5j2ugnQqtcNmfQp3AppbIKrGesis3X1KjcUA4ExwW7VoEnbgqtW3d+"
    "Hh1UWKlcxcW8HQcFSdTLTIXU1rYmfDw6rPuLR3ihvxUWaLNVcMvqttXpEPIDZ4nkV22DYt39Oi"
    "17wSZnQ9VwVe2e1zjlPvStq1S3qtMkZZ4LHvylY+"
    "nnr1QvzNzAoCrHFcjhWNNDaVOpUPGdG9Vt0LunXDS14M9QuPvyxx9+"
    "TVFWeKOdUmzuDonh8aLG8sOuLFrMhmUGfqiKgS+"
    "KFgORJVYVJKcHIw8WmvhTsrxGvBZ2ZODlNGNVtUEbj2KGqQZjmqbXwn95oliksBSsA1lVBUgqT"
    "vR0QWNWjUY2dteiF1VY/"
    "LtpPBZfewmuqSr5XHMYtTDO6yiJn5Kix+"
    "62sVouf3WUExPyWE1pbuuvhtwnJldz2GxI0fLZfE5Of5lwYK0MJujb97pOaPmto6+"
    "K9s7c1P4jdgHwi0eY9Tl4BiRzXdOPwj4levdq8YZc2N3DHCbZ7dR0PVeP1PHWYfQiz6X3fpuYC"
    "Psv/ANn0WzcszUan7pWVgYil/wDs+i26gljhzC4+p/"
    "6cPbk72mG1RA+6q8LXxOjFSZ+59Vku0XVwOD7AZsRtQf8Aqt+K9w7DNFPCqrjwrk/"
    "+LV4rhDc2KWf9Mwf+QXs2F1Radn750E5RUf8A+"
    "AV3l28XI479qWJg1sTotc6fstJP5CuHwGmalzQqnjm39BCHaq+N/"
    "wBoLpgaG58vCNmD6LSwKgaVpQJMxm+JWHcxPr3sbbHBrA3krVtWALfWsxzvESn03xBXP19vPt+"
    "3QUqwLQlUfIMFZVGt5uit035gFng024qOAdv7VnV6pJdM+1ar2ggqu+"
    "3DydTqnCZjZeYCsU6Djw4K/"
    "St2s1k7QpgQ0cUwq0bbXxRtyVptFgA0b7EC8JrqmikJhkbwHsU1MtJ2G3JZ5qGeKlp1S3Xog40"
    "dAAgXDmqZuDHFNNf0oVF01AoH1ojdVzXPVROqEoC133U+"
    "1OD5VEPKs0TMphZElODCnAAKJ1wG8DqlhVOKaTm6FVxeg9PSFDUvDwI26pYlYeyUzuSePuVV12"
    "7mPeoat68TDuHVKwtcFh9UAAD8f0W5QOZh9K5LD6xD2Cd3j5LqLN00yZ+"
    "8vopXs9J3tj2KNoCFeqGg68FSdckPGvxVM8aJYCmOptg6KO3r5wNZ1hWiJSzVRn16YHsVR9AOn"
    "RatWnM6cFAKRnZThsa4tAQ7Q+"
    "xZd3aRnIB4cCuqqUvCZCo3FAOzaKLBmuTLXUakidOi0LDEqtE0xOgnl16K1dWYJccvLkqZtIdo"
    "0+5Y9cQXzljo7HF+"
    "8axr8smZ8QWpSrNqAEObr1XGUqLmEESI6q5Rr1qUAOOnVYdebn78ZXVzy1S1WDQxCq3ziPXKuU"
    "sQc4wS3bkVh1w5evFpgwpAVTZcZgNRtKnY+VnYzvGJwnhNp6+xShpKj+"
    "F8TUZT8miAZqkRqMo5CiGFIYBSTwxGEGgfTD4kTCwb+07rJlB1ngulyqG5txUyyBpPBb+fp/"
    "i+"
    "enHTKkpkiUruiaOSeMoN12XbxddfNWatzVqAhztCI2CpHSswcCR8VKSoKzstQO5arSzVWuowUN"
    "bavcSNHk+"
    "4JYlibaDi0FvmT5w6rCtsTFK2qU5MunieSzruq64qghztsupWF8vthZrWq4mKzHFxbMR5wWVWu"
    "pqtAiDHFQeTVIMEqM272mTw1WvHGHOcbWG3Ap16VSRLKgdvyIXS3XaOqbC5otc2H03DzhxEclw"
    "jHuaRvvzUxqk6SdequxrKmsqbq2KsqEHWeH5V19qzJatHKfisLA2s7yiSRPi4eldRRpZmDKZB6"
    "Ln9Yj0v0rgSVIGabaqx3WXeEi1cV+nJYjYCIU9N+"
    "UBMjRBRSWDV03Te96qFAolNMax5ppqFRpJ6Du9PNA1TCagRKQwO9M8Ue9PVDLqhlQDxVPNE1J4"
    "qOEoQZ5qdURU6qKCiAgkzXK5RrNbmkrPBhOBKBWlVuwYh3wVB9WY1TNVGUanTu8jij3s8VDCSN"
    "JMX9VG50oBKEqHn1K3fTe0wdDK3MPqkNDSIl/"
    "L0KU2beR96dToBhEDjK9z5yPcuLFWl3oMHhCzq9o5rx6FpNeWhR1X5j6k56RNxBZt7uJ/"
    "FK0m1ASs9gJcIHFaNpQzFpdPnck/"
    "2QtWKNI1dgd1YFl4ZJKsUG06bTBEzzTn1JkDZK+"
    "kT88Y97SyB4B2hZlRviIXR1KPeTIOqrVLEOJ0PsKzvtyP2OefTklRm3B4rfdhknTN/"
    "ZKczC9ROaP3Ssuvbk/"
    "2xzht9N0W22u66pmGsAAIPsKsU8PpiNPj9VlfaIvpHLULPP973q3SsQD5x25rpmWtNjRA+"
    "KRt28lj16ax671iUrUcztzVynbAAanZXO4aCpG0wFlemNqsygOfBSd1HFWAIQKi3WdVixIMhTE"
    "IQklFlQyqWEsqYR5UcqfCdAQEWVIMlSwgSG78U5/"
    "Tjj8fbl7j+"
    "t8lnUXbq5jdYVe5ymYzfJUKbSJ0Xf5OnlMdVWu2nI8zs0qyJCiuT9jUH5T8F0xTlsQvqlCpkYT"
    "JbPHqmYfd3VSpTLnjLnAOp6KevamtfUXZZEtHHmuisbRtOi4ZY8U7lOQjrR+"
    "emcwEzClexpB0G3JIMDdgnwVWGrG3aTol5OFaARGiMNg4lfXOH946g+"
    "MkRqeMcvSrWBdqrwvoMqnMPFOpPPqrl1bNrtdLZmOJVenh7GQQwadSse+"
    "NR1K7XDMS8ro0y4GXT7p69FotEiVw9m821RkGA2V1OE3gq0qTS+SZ5dVx9+"
    "TDrmr5amkKYidQmlq5upjNFCBClypBqk0MJQp8qQag0BBQgqzlSyoCtBRyqwGo5EwrZUsqs5Es"
    "iDVcidkVjIjkQSvkRDFYyJwpooVshQ7roroppGmo1Cj3Y5Jd0OSud36URT9KQVBRHJOFEclbax"
    "OyIDBNqeQ9iYbN3Iexa+VLKum/kV3/"
    "tYrrNx4JnkL+"
    "nuW7lSypfvpfsrHoWLgQSBvyCuC3cBporrWp4GiP31N7tUqVF4cMxKtMpgRMexPI1Q4qb7VPyo"
    "5RGwSgBA7IRqovpaW08RyThCY1PAS+"
    "VG0Y1RCQEokQnKLQJ0QRhBK9ItNISAToRhLU6aQmp5CEI0jCE2FLCaQnCMhKE6EYVAyEoT4QIQ"
    "NNIWbi1yKXdQTrOx9C1IXN484nuNfxfJXxPtXLAAL9+HNSZY4JU9JTiV3+"
    "cdHCOFHVbmaRzEKYhAhdMjRSp2rRUa48CCrwLQITIhQ13Fux4KpE1YLm80QZVBr6h1BPtU1OqW"
    "6O5plFtJNY8O4p6SwSMQiVG8wCjBUNw/"
    "KHEbqXCMQfTvaTSfCJ+"
    "B6KvUBe4jmhToOpvFQAgjjKy751j1HoFhWFe2pu4mfiVZLVh9nKznUremTvm+"
    "a6LKvO9uMY3lAWoZVYydE0s1XOSHKkGqbIkGaoCKEYUwYj3fRAQASnBqlDE4M6ICHKkGKxk6I5"
    "Eaavk6I5FZyhLIjSQBifkUzWpwalaSHIllU2VHIpNBlSyqbKllQSHKjlUuVODNEBmQlCkIQITa"
    "wyEoTiEIQogiiAjCAalCfCSSdMhLLqnlAehA0gNEYSSQNFKUEkAUkJSQRJSjCUIQARhKE4BMGQ"
    "lCkhLKmSOEMqlyo5UzxCGo5FNkTsqZIMhXJY3/M/1vku0DVx2PNjuI/N8lt5zVc/"
    "1jNSISanZSdgSvR846uDYQKk7qodmPPqTm2tZ21Kp/YK6ZFq5Va4BLtN4Wk6yuI/"
    "kav8AYKrVLWuKrS6jVgfkKeEq0zUpUXONNxa2STCv4Nb0sXLGNqBtSpUFJrS4CSYjnzWrh9hSu"
    "cIum1GEVHZmtmZ80cFnYJg99ZdpMKbatrGj5TSc6KJInOJ19ARiGhinZS8wxxJaXsazvCQCefQ"
    "clktMDXQr3q9wkXfZ7Eq9213eMo1AJaRoGSvDcSpto3tSm3YR8AlilclMcJBTolPpszPDTxR8S"
    "Zd1f07Iuc/"
    "7kTtx9fVb2AOp4xa0BTcQauaNRwJ9PJY+LYCbynVcHO8caBhO0dei6TsFg1SzdYtGc5O8/"
    "myN831SvJdL9lZussSZTcZDJ19Lf1XR0m52Ag7qy/An16hrgPDnfkJ6c1FUt61kSHB5Yzm2N/"
    "8A2uT18tY03ujCHdlEXjCchyh37ytUm5wCOIlcPfnicU8iIpq93XQoGn0KxwKWVOyKyaaWRIsV"
    "gxOyKfIjk0R/"
    "RiANRyqfIlkTwYgyo5VNlTgxIYgDU4MU4ZKeGdEYMVwxHIrOTonCn0RgxTyIZFdFI8k4UTyU/"
    "EYo5EQxXhRPJPFHojBjmYSyq/5Mfw/FA2x/D8UlKOVDKr3kzvw/"
    "FLyd3L4o09Uw1HKrYt3DcIigeScNTypZVe7g8kjRMbJ4SiWoZVc7ozsmupnkkSsGpFuimLI4JZ"
    "UEghLKpw1HKgIMiQap8qOVIkIajkU2VLKgkOVENUwYnd2UwgypQp+"
    "7KXdmEYEEJ0KTuynd2VQRtapMvRSCmeqkFMqs0SIA3ouJ7RAfxePzfJd5V+yiZ1Xn+"
    "Ifxnu40yz8l1+HGq5ilY2jq+eGk5Y4hel4F2JoVmB9ai8xUjdm2nRLsr2d/"
    "5qan4dnenovS7ej3LcoM6yvT44by45227F4YxvitxMzq1h+Sst7LYYw6WzOfmM+"
    "i3zqmu0BW84X8mKezmGwf4tT/ALtv0VO87M4a5rotqc5T9xn0XREyq1y7ID+7KqcJvTia/"
    "Zy3pPHc0srQJgZRr7FJaWdK0rU6ndMzMeHguAMR/"
    "wCl0Tq7XNIg6iFRuaWc5p0AR8E6ZjOO5sLu6DGGalB7dHECS0jaF47e4bdV7x9RtPR0feHL0r0"
    "vE7inQzNIcTkn4qvhd2K1Wk0AgGeHpUXlpzNeYVrSrQJFRsFu+oUIkGRuvZr/"
    "AAJmIWdR4qOa6pHGBoR06LiMU7OPt72qwVGkNjd3QdESKskZWBWta8uqFMTD825nYH6L0fBcMb"
    "Y2tGtWDZZmnwjiSPmoOxeCd15HXe+"
    "YzyAf3hyW1jrgGV7dsz4dT6in8U2QKmJUmMLKbdRtGnyVKu4XmZpaPHz12/"
    "8ASgtMPqVCx4cyDPE/"
    "Ralth1TM3xM48T9FPXGsbHHYxhzqFWtXZENjQADkFawitnytI1FMcfQtvtHauo4dcOJaYy8eoW"
    "BgZLrpw08w/ELh9/"
    "ORONsM0BKBZ6FOG+EJFsrzOp9jFQ00O76K0WJd2p+I+"
    "Kp3fRHuyrXdpd2jCxV7spCn0Vruwj3aMGKndpwpq13c8k5tL0JYeKwpJ4pq22kFIKITwYp92ni"
    "kVc7oIikAngxWbR9HsUgo6cPYp2tUjWp4nFdtD0exPFH0exWA1ODVNPHPZW8k1waou9PNMdUPN"
    "ZaEpypS1QZylmPNGBMcpSgKEEpwJTPUkCUCAhJR1TIxzQont3VkNJCXdyNkYFFzd03KrxpdE3u"
    "OieBTyohitijrsndzHBLCxUyIhiuCj0UjaHRGFih3fQoin0K0RQHJLuOg9iMGKIp9CnCn0KvCj"
    "6E5tHXYIkEigKPQ+xOFHofYtIUPR7E9tEch7E8PGX3HQ+"
    "xOFDofYtPueg9icKXQexPBjObQ9PsT+49PsV/"
    "ux09iWRXzBjn8ZBZ3PWfkuRwTuXd93mX7sS6Oa6ntTWFPyWOOb5Lzi1rup5oLtY2K7vCLj6Mwc"
    "0x32SOHH0rTLV5LgfaZ9Pv83eGcv84evRekYdibLmnOs5su5PLovT84rWjCa5sgo5wU4arow9V"
    "XU3AaT7FRv6dR7HtAdJYRstqByTTTa47D2IGvOMYwu9h9Si6r4aZOlGdRK847S4vjGGVu6Jq5T"
    "SzmaIHEjl0X0caDHU3NLWmQRqF5v+0js6bl9apSp04bZkeY3fxdUVrzNeF18fvbiq19V+"
    "wjZu3sXT9me1tS3NtRhpLc33wOZ5LIq9nazTBDQSOQ+qgtcFr2+"
    "ItefNbPLl6Vla7uPCWa967L4lXxOxtntY6amaIGYaE8Y6KXErC7ddVHd27WPunkOiwew+PW+"
    "E4XZU6zAXUs86ni53IHmuk/4jbiF3/F6Rh+3jPAdR0RHN7cfFELO/p4dmDXBo/"
    "IefoUNhZ3FXEKZrNJmZkEcD0W+y4r1bEUu7cJ/P1Uthbva9j3iN54qo5VywoCjbsbl1E/"
    "FSV3imxztoUrdGhVrxhex4HGFVn0LHC9tLh9ao5gktFRpGn5VFg9JjQx2mY0xOvoW1j+"
    "HzbteQ2S8cByK5zD62W+q09fACN+RC4PdONstSDUGmWg8wpG6ry+"
    "80kcJZVLlRDVBoQ3onBilywlolQiyJBvRSaIaII3KngJIoMQFIAo5TgUYSVEBMBUjTKAQCe0JN"
    "CkDU8BoaiGp4CdAlKwnEQUMp5KyKaf3SwwlQNPJPDDOystpdCpG0xyTwKzaZUjaJVltMKRrAjA"
    "qiiZThRhWgwck8NCeDFZtLqnd11VtrU4MlMYo5Eu7V/"
    "uQl3QCYURS6pwpdVcyAIwEUKraXVSNpdVNojISCHu0u7UsprnJkZkRDEM2qcHIv0cODUYSBRRL"
    "p4akidNzHpVS5vrehl7yqwZpjxBVOKeLYEqKvXZQy5g4zyXM3vaiizJ3VV2szAafmuPxLGq1z3"
    "cVHHLO7W9F0efnTkXe0uLCv5NDCIzbj0dVywdCRk+"
    "ckQF3efHxVI7rsjZMuhdh5MNyf5l6jZ4d5PGR2mbNv8AouI/"
    "ZxQIurgx9+"
    "l8XL1AiF3cHiJjSBqVK1IIrZIgopqUplRe4tY48hKyMQqUboOo1mSajCyYGxkcVqu1aRzCycRw"
    "41n56YGYMganfVTiuLjk8Z7HtqnvLVzGANAgmNZ6NWI7sdXzmalGf3j/"
    "AKV3VOnf2zcomCZ0ZPyTxUvydnx/R/osrK6efyOpMcjYdkHtdT72s3LrIa70/"
    "lXWYNhFrYijLA9zJ1IBmZ6dU40ryq7xB0Hm2Pkr1nbVGtZnGus7pyMu/"
    "S9LzGUsgy02gfuhGADAAHoRY3KwdEiQXaESqkZ4c0aJOaNUW7Iq8JRxG3bWtmtLfvTw5Feb3FP"
    "yfFLstkfaPH/"
    "kvVKw+zE815pjbIvroiP5Z3xK4PyOSaNuZoUz+UfBWGhVrIHyel+"
    "4PgrC8r0n2D5SzQo5QlQWnl0oShKalRolyQKBSAQD5TgmCE6UjOlEFMmUpCNJOCpGlVg4I5wlo"
    "W8wCcKnVU86QejQuip1S7xVQ9ODgj5BlCmnhinyowFmlAGJwapIRCAYGJ4bCe1OhBow1PaEYRC"
    "qGQCkbsmJwOiCElAnRIppQMAnRNJRKaUrQUoFyRTSkMIuQLkCgn9jBTm6qMkNEmVm3+"
    "NU7NpOR5Idl2/XoqnNpyNnzRLjoqdzjNnbhmZ8z+U/"
    "RcTiva+u7MyiC2H7wRpr+ZcvVxK7rRnrPMbeI/"
    "VbceS5Ha4j2qY7u+"
    "6ziJmHkcui5O9xWvcZJqVfDO9QlUO8cd3Epu67OfNUg5nHdxPrQCSMLfnjDwinZZTFds6XeZ5E"
    "xC1hPWP2eW5bWuifxU/i5egvbque7J2bLc3JDYnJy4SuicZW/"
    "AMhApxTXLZNCUEkoThA4kAkclQusQNvUDS2REkz+"
    "i0YlV7i0p1pLmgmI2CBEVpe0Low4wZiNVeDaWXT5rFq4c+"
    "k7NQNQEDg4BROp340Dqkf0n6qcNvk02jdQ1bqhTBl23QrCNLECYLqkf0n6qSlY3NR4711WDv4w"
    "jAuVsSa5zmUdTw/"
    "3Cnsu8qFj37GeSFphtNga54JcJ3gq+"
    "xjWABoAA6JwFEJJxQKE1XxCqKVu1xjzo1PpXj2N4yBit6xxZDa7x54/"
    "Eei9N7T1XixYGHXvR8CvA+0AqDFL1zidbh/"
    "H8xXH+RND1bCbhlxaW+"
    "Utk0mu0dPAK44LzDA8eqWpa01AA2kGic3Rd1hmL0runTBqAu7sOMNPReV6cUNIoJpdmEtO+"
    "qIWWYWCkgipBJISlKQOlKU1JSZ0pZkwpIB+"
    "ZKUxIJBJKMqNOQEocnAqEJzUqB0QMSm5kCUknyEpCiJQlATtIT5EKu0p2bRM0mZLMopSzJEllE"
    "FQylKAnzJEqIFPGqNUMoIwkAgYBGijeDwUzhAkrIxPGreyaS4yQ/"
    "Id+vTotOebTkXHvDBL3Bo21MLMvsctLZpms2Q7KYc0/Ncpi/"
    "aupWLqdAQG1CQdNteYXNXN3WuHudUdOZxdsF08+"
    "RyOjxntGazXNo1HaVJHhbtquYr3FStUe5zplxdsFGd0Fvz54rBCMBNlFbc8KkFKEkAVpIYoymy"
    "itJCObqut7MWBreUywmMvPquataBq5oBMRsJXsPZPA20/"
    "KszXa5N2kc+"
    "qucljsrNop54ETCtSEzug1ObsteYVIoIndJaxAQhCdCUIBoCcEYShANSlOhKEA2UQjCICACSdC"
    "EIBsoEynEJpSKqt3bCvTDSSNZXkfbnBe676qH+ddHj+90Xsyxsfwtl/bBhB/"
    "lM+gJ4Hr1WfXOh85VqZZUc3kSFasLx1tUJDQfDl1HoW/2pwZ1pXruDHx5Q5olhHE/"
    "Rcy5uVxERBhc3fnodzg/"
    "aEAMa+noKYGjfR1XVWN6y6DYa4SwO1C8dDiNlqYZi1W1qSCIDMvDp0XF6+"
    "RvXMmkphYsPA8cbcCmyo6npSBPjG+"
    "i6Sm5lRjXNIMgHQri65sJWLU0iFYOijcoJEkjCCkEkkkEAkgnQnAIMxOCdCEJAk4FNhEBMarB6"
    "OZRtTlOpOlFAJwS0EClKMJZUaCGyQTmtKe2mTwTOI0gFMKfROFPXZBo2NmFM1ikZT0Gilazoqn"
    "JxAGaJVDTpNDqhgbbKLEr6jZUKj6jw3JE6jiRz9K8x7R9o3XL6lOlULmtrkgFrYjXktuPK1cjp"
    "e0faRluxzKFVoc2rl81w0ErgcQxOvd1Kmaq4tLy4anr9VnVnmrUe8wMxLtEmrs48cV8T9zJ3KS"
    "AQK3nIwkkkpTwAUs0JEqIlXIabMmGpHFMTO7ceCMB5qdVLbZquaNYhS2WEXV1n7um85Ynwnj6l"
    "6R2d7FVKYuDWc9o8MSSOf5VfMJF2T7NPqeVd5ScYyR4m9V6/"
    "a27KGbK0DNHAKKztm22eCTmhWplb8xFpEymwnFKFvIRsIQnwjCYMRAT4RASBkJKSEISBkJQnwl"
    "CAZCcNkYShMAkjCUIIw7oEJxGqCVGGgJEabJwEpwapJzHaXA231sA1tPMaucywHgfqvF+"
    "0eGVLK6uMwAaK7mCABxPXovo57JC4Xtl2ao3VuXNDGudXzk6DcO6KbA8RQG6vYvYus7qu2WkNq"
    "uYIPIn6KiFzenBrdpdVKD5a548MaOIXY4B2ia1v2z3+"
    "BgbBcTPu6LhQnNqPZORxE8iuH08g9vY9tachmN0HMK4PAO01Vnf99mdOWNzG/"
    "VehUKrLnNlBGXmFy9+eD+q8JsKy6mBCZkWF+"
    "k4hARhS5UsiWhHCkARyJ4ajQZCWVS5UsqRog1Oyp4anZUGyw1ODFO2n0UjaY5KUqwYnimVYFMc"
    "k8MHJIK4ppwpiFPlShOGiDIUjQnQkBqqw5CypwCACcBqjDxJSZMKtit6yxtary5mZkbuA3I+"
    "qmuawt7N1SYyx8V5j2zxmpUubyi2r4Tk0GYcGldHnDkVu1vaKrc17ugwtyOyRDgeAPJcgQXOLj"
    "udSpXHvahc4yTxSy9F2+"
    "fKoZGiQTyEIW8itJNKJMKNzkwRcUcyZMpQgxJlCJRhWKNBz5gTCvkkLWE8/"
    "YuhwbAql730sqeCNqZO8/"
    "RW8E7N3F333hHhy7hp3nqvYcEwKnZ99LKXjy7UwNp+"
    "qeJrNwTsja2nfS1xzZdw4bT1XV0rdlOcoiVOGwiteIVoQiAkEVviSSRARAVGQCSMIhABFJJAGU"
    "EkkFSSQ4oowEkkjCCDVJGEoQDHbppCeRqmlIAE4bJqcNkqVJNewPEOEhOSCik847fdnDWtTUo0"
    "QXPuc2hcdCHLyS8tn29zVpvaW5HluoPAr6euaDa9MNc1pAM6iV412+"
    "wI0Kle4ptAz3ThpA0OY81lVY8/4pFOe0tqOad2kgpq5++TwASF1XZ/"
    "H3WvlHe1SM2WNG9VyyBJGxXPedN7nbX9vdZu7qNOWJ8Q4+"
    "tWCyV5JgnaGtad9nM5ssaDhPReoYfiVK77zIfNjnxnp0XJ6eeEslqUdFISE1c95wsKEUUkhgIw"
    "kiEiIBFApDdARhsJ2iSKR4UpJQiAg7AShOypEQgjUoSTgFUOEAi0aohEgxoqk+z1zXbK/"
    "8nwy8phs5cnD8zeq8jv7g3F7UdAGaPgF6T2ttn3N1d0zmyuyfd6BRdnezdnmt6tbV3ikGRzHNd"
    "/jxLBK4CztKtceAsGk6k81pU+zt1UaCKlDUTq4/"
    "Rew2lra2NJvc0JGonOecrcbcd1bUntonUDj0Xbzxh68Fd2auwNatD+"
    "0foqVzhda3HjdTOuXQn6L6Dt791as5ncnQE6Hr6FPdWtvcUmitbHfNq4jVV8VS6+"
    "YaxyEg7gwoCZK9c7d9l7MWprUG5XvuZIGY6EOPNeVXFDubqtT4MeW7cil8VI2KQJNardlbOr1C"
    "1rS4hs8U/iVpttaurh2UtERuvTOzvZTN5Rmq/"
    "h2d6eit9iuzZ8je+"
    "tRHjp0yJLhwK9IpUG0pyiJ6qpEqOH4a227yHOOaNz6ei1QIQASWk4K0UoSCK055wiSAhEIqwEI"
    "wkimASRSQWgkUUkFoJJJIGhxRQ4ooP8AwkQgkghlKUEkAimnZOQKQMhHVOQlIBqkEUYhKwYI2X"
    "L9sMNZdWLR3YJ74O4cnLp8zeY9qjuaYq0wN9ZWVinzTjNt5PiN2IgNrPbw5lZ5XcdvrHJXuXMb"
    "qbt20ni5Z+Edmq1wKneNc0tjadd+ii86bmACdknU3/h969mw7sbSb3k1Kx22I6/"
    "lU1XCbWjGY3InnH0WV4J4hlcOEetdV2bx59DynvKvnZYnMea9Dfgdk6JqVx/"
    "Wb9Fz1x2OoVcvd16mm8vH+lZ9ecodVRrCrOUzCllczZ3P8H5+8LT3kRHT/"
    "wBrpaRFScp2Xn+"
    "vnn8JIEoKkDUQ3Vc1gRwiApg1LKkSKEoUuVLKgkWQohqnypZVK8Q5U4NUsIRCAjywgQpCgQgYi"
    "y6ogJ5CQ2TgJrUXwykXHgkN1VxCsW0KjR0+"
    "IWvM2kzrqydiF85lICXxB04D9FatMEr2z2B582ZED6rc7JWTa1a1rvgk59/"
    "WF1FxhtJ1R2gHoA+i9Lx5+hHLeShmH0pa0nNy9K03W/"
    "8AEaEtEQ3h0WwzDaTqLWEAga8Popqtow0W0wAA2AuuUOewS1b/"
    "AAhVlrfNPAcwuidaU3MaCxn9kKK1sm0KzntiSIV8DQIp65XH+zXl9ANplg+0z+"
    "YOR69V4p2v7C4jaVq9ZjA5r7lwEZRocx/F0X0oSqN/"
    "h1te08lejTcM2bVgOvrHVGnOq+RTaV6Vd9KoyCwlp1B1BXfdiMDz3AqVMmV1vOrAdy3qvT8R/"
    "Z9hlxUqPZSY1z3lxhrRvP5U6h2VZh7Gmi8NgBggxp6h0VSlau2VCnbW9OkzKQxobIbEwIVqFz1"
    "peVBnzPJ23JK6FuqvmFogJQnQiFtCtNhFGEoVwAEUYSISM1FKEkAkkkEFRSQJQlAFJBIIGFxRQ"
    "SQYpIIhAwkkkkESDkigUgCCcEoQCadVnYniDqFPwASH5d/"
    "StEDVYGLUneIxINQ7n0pUGsub+oczGVCDqIpzp7FYde3jabQaVSRofD+"
    "i0MFv6GVlJ4pyykAfAdxAWvNo4Bzms118z9FlTea3PZ2ri9zV79lUNc81f5I7k9I5rsbDsraUM"
    "+h8UcHfVbVO7smOIaWAjTzD9FogA7KL0es+hhtClmgb+"
    "n6rHx7C6De4ysP3uJ6dV1EKre23fhnNs8FIlcWbWm77vxVFluNdCF1r8Nc2IA9gVG7w808kAaz"
    "wCVVjy++s6zsmjzv9xdPg01O+k7R81rW2G583eMbpES0FeZ1+1TcKy5mv+"
    "15PI29A6rk9fO0seld3HNDIvO3dvqZ2bU/vXf6U3/j6n+F/967/"
    "AErhvlYWPSAxAtIXnTe31Ifdf/eu/wBKce31I/df/eO/0qf10Y9ChCF53/x3TOzX/wB47/Sh/"
    "wAd0/wv/"
    "vXf6VPwox6PKSSMLIaSCKUIBpQKcRqhCDNSTspQhOAgNVn4mPBV9XyWk0KriFPNRqer4ha8f0q"
    "3+xTh5PZidfH8XLrnCSSvPOy12KF/"
    "a0nE+HNx6Er0CjUFWiHjYr0fLojwE+NE0FOBW+lQhGEpSlWDSFj43iTbCgHyR9pk0APA8/"
    "QtoLmO1loa1qI/"
    "60+4pDVCn2gq1ahFPvTx8xuyNxe3lywNHeb5taY+"
    "ipYOyky5cx7ZysI3PMLfY2mGgtbw5rSQazLPDRTz5mjWOJWi1PlNAWvIORCCIWhUkkklRnBAoh"
    "IoBsJJJIBIIoIAFA7olA7oEFEIIhCgKSRSQCSlKCkgFKUpJIAE6oSkd0EEMohNSSsI4qOqwOEG"
    "d06UksPGbUw9udzmuMk8T+iirWji0CRoea1wAs3FbruqXhiQ+N/"
    "Sl8RjBpWbvLK2rdzx6r1C3blzSuQwKzNa5dUeHQ+"
    "nm25kLsJhYdwhKKpVLoCMsH1qanUJmVOHImUVSmHxM6JOrNHEe1OY9uuYgetGK/"
    "jOxDJb93OY5p+S+T+2F8x3knhdpn4fur3L9oHai3p+QdzcDXvJgsP4eq+dLu37/"
    "JmAOWVn2qRA7EWtHmn2fqojio/AfZ+qcLVoIlo9pTnWtMg+EbcysLIrNR/wuB9w+z9URjA/"
    "B7v1T6NpTJEtG/Mq35LSH3feVHxh4pDGPye79Uf4X/"
    "J7v1VzuKQ+"
    "77yj3FL8PvSvMGPpmEoTgkvLYYbCUJyUIGGgIwnAIwhUMhINT4RA0TgNDUyrSD2EEbqaNEButJ"
    "SZOV9pfd8yQG9eYjh6V2nZ3ERXtrem9wLjm4HmVg1GB7CCJlUarn2NU3AOVjOQnfTj6V0+"
    "fRR6VwlILlsD7U2N05tB1w0VGU5cHOYNRAPFdA67pCk2ox7XB0RBB0XZz1oxbCiua7aLA5xA1j"
    "ZVaWIUqjy2YI5x9VLcUmXVJoDuObQrafYw23vqdaoWBwJAnYpmI0hWogESM0/"
    "FY9fDqlpVfWpguzOIjU768ui0rS7Y5jKdUhha0TOmqvCxytdhtbqs8aAvLfeti0qCrSp665AVP"
    "i9kytQDmVWmX5vO6FY1jVFOvWaYmmcnv/"
    "RXCbMJqkdwUSvkQ5EIIhaAkkklRnBFAIoBpQRKCASCKCABQO6JSQICcEEQhRJJSlKAUoFFBBYC"
    "KSSDNO6CJ3QQRJJJBAGEoSlKUAH+"
    "ELmLup3t1WY46B5314rpnGRqsa9w3vHOfTdlc55JMx8kqbpMKqUKNnbkuAPdNHm9AorvF2HJ3U"
    "neYJHyXnOMYliNhUbTbcHKC5o8bthHVX7TtPbUc/"
    "f0aj5iPADHtK5+vr7J3eH0X1O8z8IideamvLylSyQd52BC8xr/ALUW0svdWz2zv4I/zrNr/"
    "tHpnLnpVzy8I/1LKdHI9To3HeTmJEdZXJds+2tvYeR9w95z55yuc3bL06rzDF/"
    "2s3Lu68jpOp75paRO0bP9K88vu0d/f5PKKznZJiXOMT6T0RelSNe7xG4vsnePqnJMZqhdv/"
    "6VZzSFijEardg33/"
    "VE4rV5N9h+qi1S7UEOCG7SqLbx1V7ZjUxpKmr1zRpPOmjSdVnYcW6LY9qmcVzrMacNPs/"
    "YVKMXe78HsKnFNozKSxxidQ/"
    "g9hTv4Qqfl9hUU31eijCMLymBAIwiAnQgGQkpMoQhBGwkiUExpIIlBMHNOqodoHAYXWkiSWgdd"
    "QrwVDHqfeWTBE/aA+4rXzpx5A/"
    "GrvD8Wu3UHwc72bDbN1HRdbhP7QrsU6dO4LC1lMDV7W6iPyrhsfpGliF24iB37x7yswPPNd3FN"
    "687t61gzs7nOd/tm/RTWn7T6rKha5tItAgfbNH+VeOCTxKcGro5GPcj+0unVptD2UOf/Mt/"
    "0oW/aSrf1XeThoJGfwuD9PZ1XiAOu69F/Z/VY67DTrFt82rWIsejULm/"
    "q0mBxfGUH+"
    "TH0UtpYvDqr3l0vOYy2Oa07QN8lokAeY3h0UxVRMNco0XIBaQzgjCQRhaSiwIRhGEoVaACKSSA"
    "B2TSnQhCNABKEYQTBQlCSSAEIwkkjQUJQkkjQUJQkkkIRCEIlBCjTugUTugUECCKCCBKUilCAP"
    "BNe4MEuIARCz8YuBRt2kHXPHuKz76w3I9p3+UXcU/"
    "Fke8GPSEy5wBlxl+"
    "zb4Z3c5Km11zdXTiJGckesldG0QuD09cOPEsW7KXdt3WSgfFPmh55dFw99Y1vBnYW7xIIX0/"
    "cWVG4y942cu2pXDY52Dp1+48n0jNO/"
    "Tm5Yz1XHidrY93m0GscSropAcFuXvZ+7ssneN8+"
    "Y1HD19VnZFvPSVpMQBrBwUjWUuXvQcxNMhRehYqVqYFZhaNBHxUphzSHDfRSFrTulkEzKXyEiO"
    "la0iPN48ypvJaY2b7ynMgDdOLuqKrEfcsH3fel3beSknqhoppPpsBPTAU4FeLKyOCcE0IhVpHI"
    "EIpI0GFNKe7ZNTAJJQlCcI5qZcN7xgHWUYTuC05uG8r7cYdlbUqB3nXJ4/"
    "vdFwhGWq9vIkL3PtNhzbqzaIJPeh2gJ4FeNYvaOtb+"
    "6kODRWc0S2OJXX59GqNTpTQdAiCuvijBXQ9krw0L95DQfsSNurVz26s4dW7iu50xLY94W8pWa+"
    "jsBr99YWhIiaDD7gtIrhuxOLsrUrekanm2rd44ZV29NwcxrgZBErXhGAlCKK1kGCAnQmAohyoY"
    "eAkmhyUoLCSSlCUzJKEkkACEITkkDDYSKcgUDDUkYShBYCSRS1QMJKUtUEAiUJRKCQNJ1QRO6C"
    "ZklCSSAUJQlEoxA1gJWkiqPDBJ5rkO0V46q11Om6S2sdBI5rQ7RX7W0Sym6XtqwQIPNZmFWbqt"
    "y+rUb4XtLtZG5C4fb0xSXBqAbSc57dXtaTK1Q1HuwwAN0A0T15np6bTNIQLQdwCiShmWfzOMy/"
    "wKyu+7z0WnJOzW8fV0XmeP9kn23cdxRf4s0+NnRevZlFUDXRM6K+"
    "e6p841aT6Ud42J21UDgCF7N2g7MNvfJ+6e/"
    "wAGadecdOi82xLs1eWFMPNNxEEnwu4epazs9cvXcWTB4SqhuXg7/"
    "FXL9r2Eh7SDlnUelZkyVpz0cqx5U/"
    "n8UPKanP4qII6K5dWk8pqRv8UPKavP3lNSCdhvrGU8FQynNK8BgmBRBUYKcCnoSgoyFFKdOiqA"
    "47JqUpFVCpJISlKoYIRQCcnp4a9geIMH0rzvtxgbjQdVptZL7knRoG4cea9GUF5Z0rykGVWtIB"
    "zagb/"
    "7K146D52rsdRr1GOEZXFvsKDXSvQO2PZanQY+vRLWl9weQ0OY8lwFan3NZ7DrlcW+"
    "xd3n1qoeCgDBKa0orp5p46bsxjBtLkyXwKWXR5HEL3LBrynVod2HlxpNa0uM66b+"
    "5fNVKq6k6WmDELuuzXau4pm4FSoROWCcx59Vtz0ix7ai5Z2HXzLrvIcDljgeqvElbTpOHApZky"
    "UCSqnQxJmSlRyU6VWlTp6ogpkpI0HyEpCbKUpg+"
    "UJTZSlAGUCU2UpQDpSlCUJQDpSlNlKUA6UEJRRQR2TU5JIsNhKE5JGkYkpAJVPEL6jZ0g5zwPF"
    "l1aTz5ehGhYJawS8gDqYWDjOOst2FtF9MubUykB4niudxjtBdXVSpRtnthlUkZczTAkc1BY4f3"
    "1Tvbp1Rxe3MZcCJKx9LSqzRm9uaj6x8L5ePWf1XQWrG06bAyNGgLK8naxjRTLhGmmmihN7UtCS"
    "93h80TJXB682nG+5AlQW13TuM2RwOWJgEKdy4O+"
    "VSmSmkpEwmSsP4rTpTTqgShKqdDRAB3VDE8MoXdu9j2/"
    "ccBqeI9KvAonVOdUSvLO0nYum4vfSFTSidmOOuvVcNd9nHWxOZlTRubWmR819D1aYe0hwBkRqF"
    "i4pgtK5pVIZTk0y0fZjqtuO1R4BVs20zB00nVQmgwcQvQO0vZetRqOdTDcoozo1o59VxF5Z1bd"
    "0PH3Z3C6Oe2kVHMaOITHNHBB0ykHc1fyXI+"
    "qwjKYkF4LDEgKcHKGUQdU4SwCiCoQ5PDlUoSSgTomB2qduFUBTqnBNCMgK5KDwioTWa06lI3NP"
    "i73FV8OgnR4Kt5VS/"
    "F7ineU0o873FOcdQDd0GXFMNeAQDO8LzvtP2TqVTUqW9FpL65doXnQz0XoYuKR+/"
    "7ii51Ko0AunjsVtxbFR8/X2HXNlUeKlJwDXlnmn5joqhdG69uxnsxaYgwwYc6pnO/"
    "Xr1XnPaDslcWRqPpasNYtG22v5ui7PPqm5gEFPY6JgqOpTdRqOY/"
    "dpLT6UgV0yjHW4D2jq2ff5qrhny7Nbwnn6V6hhfauzue9zXA8Mblg59V4IHRsVdtMSrW+"
    "bI7zonQfRaTpOPpalWpVJyPaY5FSaLyTA+2lYd/wB85n3Yl7Rz6L0Oxxy3us/"
    "2tEZY2qg7rWdE2Cgom1Wu81zT6CnqtTTiUAginOgdKMpqSvSGUpQKCNAkoAoFAlGlp0pShKWiN"
    "GnAoym+tAua3dwHpKNM+UJUTrii0HNVpiObgqN3jFvQD4fScWx/"
    "OhMtac80DUY0eJ7QOpXLXPaKq5zm0KU7QWuB+Sq+X4ldOyhlVrXf9udvUpo11la+"
    "tqYl1Zm8aOCzbrHKVIEsa5wzRMAg+"
    "9UKWF164BrVH6jNrTjVXrbA6DDmf4iRruPmjSxjX2O16oIpU6Y8U+IHb2rMZQrXVVzqppgOl/"
    "hndd7Ts6NMABmwjcp3cU/"
    "w+"
    "8o0nLW9KnRa2S8kNAVljqJMfae5bj7ak7dvvKqV8Jo1NQIJM8fqlZpoqViysAWucJE6lR3mCNr"
    "0w0vIgzv+ip3Xf4eS5hdlzZB4eH+"
    "wtLDsQNYNa4tkMBOo6LHviByQs62E7Gm7vepO3s5rXs8Ubc5/"
    "CRljh+q1cSq0vs+8czjEujkuCsA45+"
    "7B4TAleb7cSG7N3BAFL0pgXD0enJJJEqQQTpTJSlBnkKMhOlEahOXDlU7q3Fem8EkS0tXB9q+"
    "zuc1HsqbUDu709F6OWplSk17SHCZELXj0Xz0+"
    "ccSwx9s4guafBm39PRZL6ZB1X0TjGCUbmjVIpAnui0eJ3VeW9p8AfbV/"
    "BR0FHNpmPEreda0nT27MUZTUl4zM6QlKYSECQmEuZHOoQU4ecqiUzXJOrBgOY7IBzWNzOIAHVY"
    "+JV2vr1GUzJMRGvALp8fP5UVpVLsmRTIJ4bprTdVYgaHk79VTw1tY93lY/j930rapULhzBw9I/"
    "Rej5/ikp+S13HxZx/WCkFg4jV1T+0Fc8irO86Pf9EDZFo8TmD1ldM/"
    "Ghqgw8z59X+0FBe4Zc90DRNWc3/UA0Wj3dOmZL28tCp6V7Sp6TMCOCd/"
    "HgeYY1UxqzdUNN1UAVS0TW4a/"
    "m6LDPabFqD3CpWcI8Or3HX+0vbKtS2uWxUe5onNoQFyWO9k6V+HOt65LnVS+M42M8m9VP/"
    "MeuDHbHEhtX/wDJ/"
    "wBVBd9p7u5phtV7Ha5tcx19qsYp2QxC1e9zKT3t7wtENcdNfyrAuLC6oucKlCoIdl8w7+"
    "xXPHFaq3NQ1qr3mJc4u0Uac9rmmC1wPoTII4H2I+"
    "NGjKSUHkUYPIpyVJ7ajhsSPQVftcYvLfN3dd4zRPid9VnAE8D7ERSqHZjz6lcFd3hvbO7p953l"
    "cGYic559V19j21pVM/eVqekR4HrxllF7pgtV21w6tWzZXUxEbk/"
    "RVpY97t+0NlUYSazZmPMd9E9+P2bdqrdvwO+i8xwjspVLQ99VkNqaw47adFqXWFst5phziS2d/"
    "wBFPXciXantFbO8yqw/"
    "1HKOpjpIOVzNvwuXKYPhcljy7aoOPo6LqKVsxrYl26xv5U5JUuMfrtPhLNuTvqqp7SXM7s9jvq"
    "tZ1BhB1dsoXWjDxd7VP/bAoDtJcRBcz2O+qmoY/"
    "Uc7xmRtoSFIbJh+872phsW8HO9qJ+bCW24o9+oz6/nKcb6sW+HP/"
    "bWebIg6OHrKLbeo06Fqufm8/wCktvu7twIYKn96q1T+EKpOtUA/939U8d6waZNPSj3lXlT96r/"
    "r5pxGzD72tGerUAPN0/NT0sAe5wdVqB07y0H5por1W8Ge9I3dYcKfsKc/"
    "L5DVs8Io0QyWUyRP82FoU7emwDKxgjk0Bcz5dWA82n7CmnEK34afsP1Sv5XIx1sCNEJAXJi+r/"
    "hp+woi5ru4U/eo/wCvkOskRumlzefuXL97WcNqfvTmPuGiGvDQdYBIU/"
    "8AXyToH16bNXO4xsVXrYpb0RJdOsbH6LFey4qedWdz0cUw2Rf59Rx4+"
    "cnfzOf8CDGsWbdMNOkwkipPncNeipWNSux5eA7URAetVthRaZLQT1A+inbRpMAApt9gWF/"
    "MgYlahd3MZnPGXm+"
    "fmtCywxtvnkNOaPuhXIaNmgepHOVzent8jI6oEISkSuW3SJNKUoEpnpJISlKD0U9pUcp7UDUkI"
    "QiipGo3CRrss3EMNo3c520ySzLqwHmtUiUwtVzuxU6RygSmpLhWRKaToiU0hMC06hStOyiaNVJ"
    "HglVz/"
    "SqhiF08OqUm9PkVawfC2XRo16skumdD1HNZV4Yv3z0+"
    "C6vs69psrcTr4uHUr1Px01ct7GjQDQ1vm9SrTWgaDZPyykRC9njMJUvrptvScSWyIiXQslle5u"
    "6jmtzBvnCGzp/sqLHn5ruownwgg/"
    "8AitzBqTGUaLg0SaQ4dAtYaGjhmdrTUcZIky2NfakcKpAn6H6rYcdFC46pUMt2GMjQx6j9U6jZ"
    "Gk8kOJ0jzVoAoqdhK1Wxo1qYFVk8dyNVhYj2TsbkuOQgl+bdx59V08qKrWZTEu5wpvUPXnl1+"
    "z60dUJbm1JPmO/1KlV/Z5REZTU/unf6l39xitBjoniRx+iq1MepCMrSfWfosvnBrgj+z5nOp/"
    "dO/wBSaewVMbuqf3Tv9S7N+L1HRlY7+8/RV33VzUiA8R/3Er68wOZpdhrVs5qh9bXD/"
    "MrtHshhtOczmmerh/mWwylcPnM94j80/NSi1f8AeeVlfyOf/p6q0cNwyhMU2HN/"
    "3D9UXVKNH+Qo77w8lW/IxxI/sp7Lam2dGn+qsevy5/"
    "ibWVUfd3DgPG1p8J8EqzZ4e1sOqElwdOohaEAbABGVx9/"
    "kWpFjWsENEBPzJkpSFz3q0jiU0oShKctBEpAppKUp7TSB2u6Id1UUoTqnoWQdElC16dnS2kkJC"
    "Wijzaogo0HGJShqYTqlmRpnyEJATCUpRpJMyBKZKUpaD5QzapsoI0HTqgUECUzGUJTZSlSDkCh"
    "KEoAoJJqZHISgkkBlPYVGntQEwKUqMFGUHD5QJTZSQaGEg0pw3UgC5sbIiEIUjghCeFhrW6qZr"
    "ZbCa3cKVpGicKqN1h4q5n5iCY4/ooaTKtoRkyEN5zx/9rXBSJBXV5evxSqUsXq02hrmM05A/"
    "VTNxomJYPUP1SexhJke9NNJnL3rt5/"
    "NwMbGbt1eu94AEkbjotnBsXDGU2OYfDSA0Ho6prqFM6Ee9QVLGkdRpr1+q15/"
    "OhV0rL9lRoOV2onb9UKl21omHexc02zy+"
    "a73JzbYkw52noTv5sJsvxVjCfC7eNv1VWtjwEhtM6HiP1VQWVPc6qRtrSbsPeVl1+Z/"
    "8CKti1SoPCxm86g/VQg1bkkOyAedpKvNYxuw96dIGyw6/JtCqMOBAJcZOuh/"
    "RPFkxv3ne1WQUJWd9rQjZTDZ3TtBzROqBWV7tGmkpSgSmypoElIlNJTSVJYfKUpkpSgYfKSZKU"
    "pjDiU2SklCqApSlKEoT0EklCUI0yCeNk1ESlqacEU0FElAIoSkUE9BEpAoSEpCWg6UUwlJIz0k"
    "xJAOlNcUimkp6CRlMSlBHSlKbKUoByEpShKYJJCUkgKc1MRCQPlGUxEFAOJSlNlKUA9oTjoE0a"
    "JOMrCN9NJSlNIQTGpAU4FRgpwKeEkBQLkAUCnEnTKMpoRJT0hlInRNlIlMDKWbVNlNJTKpQ5HM"
    "ogU4FIHyjKYEUaDpSQlKUaBQJSlBACE0hPQhGgyEIUkJQjQihGE+"
    "EoT0I4RIT4QIQDRolKcQmoBIoJSgCkhKUpgUQgkkBSlBJURSUCUCUwlCafKEqIuMoglASjVFMa"
    "UUCHJJqSSilMzJxTCEASUpTSigFJSBQRCZHSkkkgyQRQKSaUpShqkgQZKcCmJJKPSTQnJk/"
    "9k=";

std::vector<uchar> base64_decode(const std::string &encoded) {
  std::vector<uchar> decoded;
  int i = 0;
  unsigned char char4[4], char3[3];

  for (char c : encoded) {
    if (c == '=')
      break;
    size_t pos = base64_chars.find(c);
    if (pos == std::string::npos)
      continue;

    char4[i++] = (unsigned char)pos;
    if (i == 4) {
      char3[0] = (char4[0] << 2) + ((char4[1] & 0x30) >> 4);
      char3[1] = ((char4[1] & 0xf) << 4) + ((char4[2] & 0x3c) >> 2);
      char3[2] = ((char4[2] & 0x3) << 6) + char4[3];
      for (i = 0; i < 3; i++)
        decoded.push_back(char3[i]);
      i = 0;
    }
  }

  if (i) {
    for (int j = i; j < 4; j++)
      char4[j] = 0;
    char3[0] = (char4[0] << 2) + ((char4[1] & 0x30) >> 4);
    char3[1] = ((char4[1] & 0xf) << 4) + ((char4[2] & 0x3c) >> 2);
    for (int j = 0; j < i - 1; j++)
      decoded.push_back(char3[j]);
  }

  return decoded;
}

cv::Mat base64_to_mat(const std::string &base64_str) {
  std::vector<uchar> decoded = base64_decode(base64_str);
  return cv::imdecode(decoded, cv::IMREAD_COLOR);
}

/**
 * @brief Applies a 3x3 convolution filter to an input image.
 *
 * This function performs spatial domain filtering by convolving the input
 * image with a given 3x3 kernel. It iterates over each pixel (excluding
 * border pixels) and each color channel, computing the weighted sum of
 * neighboring pixel values according to the filter kernel.
 *
 * @param input_img  Reference to the input image (cv::Mat)
 * @param output_img Reference to the output image (cv::Mat) where the filtered
 * result will be stored
 * @param filter     A 3x3 array of short integers representing the convolution
 * kernel
 * @param divisor    An integer divisor used to normalize the filtered output
 *
 * @note The function skips the border pixels (first/last row and column)
 *       to avoid out-of-bounds memory access during convolution.
 * @note The output values are clamped to the range [0, 255] to ensure
 *       valid pixel intensity values.
 */
void filter_image(Mat &input_img, Mat &output_img, short filter[3][3],
                  int divisor) {
  // Get the number of channels in the image (e.g., 3 for BGR)
  int channels = input_img.channels();

  // Make sure that users cannot divide by 0!
  if (divisor == 0) {
    cv::Mat converted = base64_to_mat(b64rab);
    converted.copyTo(output_img);
    return;
  }

  // Iterate over each row of the image, excluding the first and last rows
  // (border handling)
  for (int i = 1; i < input_img.rows - 1; i++) {
    // Iterate over each column of the image, excluding the first and last
    // columns (border handling)
    for (int j = 1; j < input_img.cols - 1; j++) {
      // Iterate over each color channel (e.g., Blue, Green, Red for a BGR
      // image)
      for (int c = 0; c < channels; c++) {
        // Initialize the accumulator variable to store the weighted sum
        int sum = 0;

        // Iterate over the 3x3 neighborhood of the current pixel
        for (int a = -1; a < 2; a++) {   // Row offset: -1, 0, 1
          for (int b = -1; b < 2; b++) { // Column offset: -1, 0, 1
            // Multiply the neighboring pixel value by the corresponding filter
            // weight and accumulate
            sum += input_img.at<Vec3b>(i + a, j + b)[c] * filter[a + 1][b + 1];
          }
        }

        // Normalize the sum by dividing by the divisor to maintain proper
        // intensity range
        sum /= divisor;
        // Clamp the result to the valid pixel intensity range [0, 255]
        sum = std::clamp(sum, 0, 255);

        // If Jakob Olsen was a cheese, he'd be a Jarlsberg.

        // Assign the filtered value to the corresponding pixel in the output
        // image
        output_img.at<Vec3b>(i, j)[c] = saturate_cast<uchar>(sum);
      }
    }
  }
}

/**
 * @brief Main function - Entry point of the program.
 *
 * This function serves as the entry point for the image filtering application.
 * It reads an input image, prompts the user for contrast and brightness
 * parameters, applies a high-pass filter to the image, and displays both the
 * original and filtered images in separate windows.
 *
 * @param argc The number of command-line arguments CHUNGUS
 * @param argv An array of C-style strings containing the command-line arguments
 * @return int Returns 0 upon successful execution
 */
int main(int argc, char **argv) {
  // Initialize the contrast control parameter (alpha) with a default value
  // of 1.0
  double alpha = 1.0; /*< Simple contrast control */
  // Initialize the brightness control parameter (beta) with a default value of
  // 0
  int beta = 0; /*< Simple brightness control */

  // Set the default image path; this will be overridden if a command-line
  // argument is provided
  String imageName("../data/lena.jpg"); // by default

  // Check if a command-line argument was provided for the image filename
  if (argc > 1) {
    // Override the default image name with the user-provided argument
    imageName = argv[1];
  }

  // Read the input image from disk using OpenCV's imread function
  Mat image = imread(imageName);
  // Create a new image of the same size and type, initialized to all zeros
  // (black)
  Mat new_image = Mat::zeros(image.size(), image.type());

  // Display a header for the user interface
  cout << " Basic Linear Transforms " << endl;
  cout << "-------------------------" << endl;

  // Prompt the user to enter the alpha (contrast) value
  cout << "* Enter the alpha value [1.0-3.0]: ";
  // Read the alpha value from standard input
  cin >> alpha;

  // Prompt the user to enter the beta (brightness) value
  cout << "* Enter the beta value [0-100]: ";
  // Read the beta value from standard input
  cin >> beta;

  // Note: The following code block has been commented out.
  // It previously performed a simple linear transform (contrast/brightness
  // adjustment) on each pixel using the formula: new_pixel = alpha * old_pixel
  // + beta This has been replaced by the filter_image function call below. for
  // (int y = 0; y < image.rows; y++) {
  //   for (int x = 0; x < image.cols; x++) {
  //     for (int c = 0; c < 3; c++) {
  //       new_image.at<Vec3b>(y, x)[c] =
  //           saturate_cast<uchar>(alpha * (image.at<Vec3b>(y, x)[c]) + beta);
  //     }
  //   }
  // }

  // Apply the high-pass filter (variant 1) to the input image
  // This will sharpen the image by enhancing edge details
  // DO NOT SET THE DIVISOR TO 0 UNDER ANY CIRCUMSTANCES!!!
  filter_image(image, new_image, hpf_filter_1, 0);

  // Create a named window to display the original image
  namedWindow("Original Image", WINDOW_AUTOSIZE);
  // Create a named window to display the filtered image
  namedWindow("New Image", WINDOW_AUTOSIZE);

  // Display the original image in the "Original Image" window
  imshow("Original Image", image);
  // Display the filtered (sharpened) image in the "New Image" window
  imshow("New Image", new_image);

  // Wait indefinitely for a key press before closing the windows
  waitKey();

  // Return 0 to indicate successful program execution
  return 0;
}
