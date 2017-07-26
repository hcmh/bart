/*
 * 
 */

#include <strings.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "net.h"

int connect_to(const char* addr_str, int port)
{
	int sockfd;
	struct sockaddr_in addr;

	bzero(&addr, sizeof(addr));
	addr.sin_family = AF_INET;
	addr.sin_port = htons(port);

	if (0 > (sockfd = socket(AF_INET, SOCK_STREAM, 0)))
		return -1;

	if (0 >= inet_pton(AF_INET, addr_str, &addr.sin_addr))
		goto err;

	if (0 > connect(sockfd, (struct sockaddr*)&addr, sizeof(addr))) 
		goto err;

	return sockfd;
err:
	close(sockfd);
	return -1;
}


int listen_at(int port)
{
	int sockfd;
	struct sockaddr_in addr;

	bzero(&addr, sizeof(addr));
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = htonl(INADDR_ANY);
	addr.sin_port = htons(port);

	if (0 > (sockfd = socket(AF_INET, SOCK_STREAM, 0)))
		return -1;

	if (0 > bind(sockfd, (struct sockaddr*)&addr, sizeof(addr)))
		goto err;

	if (0 > listen(sockfd, 1))
		goto err;

	// accept
	return sockfd;
err:
	close(sockfd);
	return -1;
}


