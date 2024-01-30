#include <stdio.h>

int main() {
	#ifndef ONLINE_JUDGE
	freopen("input.txt", "r", stdin);
	freopen("output.txt", "w", stdout);
	#endif
	int a, b ,c;
	scanf("%d%d   %d", &a, &b, &c);
	printf("%d %d %d\n", a, b, c);
	int *d = &a;
	printf("%d\n", *d);
	*d = 3;
	printf("%d\n", *d);
	printf("%d\n", a);
	return 0;
}